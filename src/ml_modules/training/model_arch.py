import json
import sys
from os import path

import torch
from torch import nn
from torchinfo import summary

import torch_geometric as pyg

class MultiGCN(nn.Module):
    def __init__(self,
                 graph_dims,
                 dim_node_feat,
                 dim_node_hidden_ls,
                 fc_hidden_ls,
                 use_ogt=False,
                 feat2fc=False,
                 node_feat_name='x',
                 dropout_rate=0,
                 dropedge_rate=0,
                 dropfeat_rate=0,
                 dropfeat_layers=[1],
                 dropnode_rate=0,
                 conv_norm=True,
                 fc_norm=True,
                 global_pool='mean',
                 debug=False):
        '''Instantiate all components with trainable parameters'''

        self.debug = debug
        if debug:
            torch.autograd.set_detect_anomaly(True)

        ### save a copy of arguments passed
        # for cloning the model later
        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        self.graph_dims = graph_dims
        self.n_graph_dims = len(graph_dims)
        for dim in graph_dims:
            if dim not in ['contact', 'backbone', 'codir',
                           'coord', 'deform', 'pae']:
                raise ValueError(
                    f'{dim} is not a valid graph dimension (valid options: '
                    f'contact, backbone, codir, coord, deform)'
                )

        # tunables
        self.node_feat_name = node_feat_name
        self.use_ogt = use_ogt
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.dropnode_rate = dropnode_rate
        self.feat2fc = feat2fc
        self.global_pool = global_pool

        # internal parameters
        self.training = True
        self.debug = debug

        super().__init__()

        ### INSTANTIATE CONVOLUTIONAL LAYERS
        if not isinstance(dim_node_hidden_ls, list):
            raise TypeError(
                'dim_node_hidden_ls must either be an empty list or a '
                'list of ints'
            )

        # determine whether bfactor and pLDDT should be included
        n_add_feat = 0
        self.use_pLDDT = False
        self.use_bfactor = False
        if 'pae' in graph_dims or 'contact' in graph_dims:
            self.use_pLDDT = True
            n_add_feat += 1
        if ('codir' in self.graph_dims
            or 'coord' in self.graph_dims
            or 'deform' in self.graph_dims):
            sum(dim_node_hidden_ls) * self.n_graph_dims
            self.use_bfactor = True
            n_add_feat += 1

        # first dimension: one-hot-encoding for residues (20) + b-factor (1)
        dim_node_ls = [ dim_node_feat + n_add_feat ] + dim_node_hidden_ls

        self.conv_block_list = nn.ModuleList([]) if graph_dims else []
        for _ in range(self.n_graph_dims):

            mods = []

            for layer_idx in range(len(dim_node_hidden_ls)):
                dim_input = dim_node_ls[layer_idx]
                dim_output = dim_node_ls[layer_idx + 1]

                # normalization
                next_idx = layer_idx
                if conv_norm:
                    mods.append((
                        pyg.nn.GraphNorm(dim_input),
                        f'x{next_idx}, batch -> x{layer_idx+1}'
                    ))
                    next_idx = layer_idx+1

                # dropfeat
                if dropfeat_rate:
                    if (dropfeat_layers == 'all'
                        or layer_idx+1 in dropfeat_layers):
                        mods.append((
                            nn.Dropout(p=dropfeat_rate),
                            f'x{next_idx} -> x{layer_idx+1}'
                        ))
                        next_idx = layer_idx+1

                # convolution
                conv = pyg.nn.GCNConv(
                    dim_input, dim_output,
                    # GATConv parameters
                    # heads=3, dropout=0, concat=False,
                    add_self_loops=True, bias=True
                )
                mods.append((
                    conv,
                    f'x{next_idx}, edge_index -> x{layer_idx+1}'
                ))

                # activation
                mods.append((
                    nn.LeakyReLU(),
                    f'x{layer_idx+1} -> x{layer_idx+1}'
                ))

            # jumping knowledge connections
            feats = [f'x{i+1}' for i in range(len(dim_node_hidden_ls))]
            mods.append((lambda *x: [*x], ', '.join(feats)+' -> xs'))
            mods.append((pyg.nn.JumpingKnowledge('cat'), 'xs -> x'))

            self.conv_block_list.append(
                pyg.nn.Sequential('x0, edge_index, batch', mods)
            )

        # total feature size accross all layers and all graph dimensions
        total_node_dims = sum(dim_node_hidden_ls) * self.n_graph_dims
        if feat2fc:
            total_node_dims += (dim_node_feat + n_add_feat)

        ### INSTANTIATE LINEAR LAYERS
        if not isinstance(fc_hidden_ls, list):
            raise TypeError('fc_hidden_ls must be a list of ints')

        dim_linear_ls = [total_node_dims] + fc_hidden_ls + [1]
        if use_ogt:
            dim_linear_ls[0] = dim_linear_ls[0] + 1

        self.n_linear_layers = len(dim_linear_ls) - 1

        self.fc_block_list = nn.ModuleList([])
        for layer_idx in range(self.n_linear_layers):
            dim_input = dim_linear_ls[layer_idx]
            dim_output = dim_linear_ls[layer_idx + 1]

            fc_block = []

            # normalization
            if fc_norm:
                fc_block.append(nn.BatchNorm1d(dim_input, affine=True))
            # linear connection
            fc_block.append(nn.Linear(dim_input, dim_output))

            # for non-output layers
            if layer_idx != (self.n_linear_layers-1):
                # dropout
                if dropout_rate:
                    fc_block.append(nn.Dropout(p=dropout_rate))

                # activation
                fc_block.append(nn.LeakyReLU())

            self.fc_block_list.append(nn.Sequential(*fc_block))

    def forward(self, data_batch):
        '''Make connects between the components to complete the model'''

        # get node features
        bfactor = data_batch['residue'].bfactor.float()[:, None]
        pLDDT = data_batch['residue'].pLDDT.float()[:, None]
        x0 = getattr(data_batch['residue'], self.node_feat_name).float()
        # res1hot = data_batch['residue'].res1hot.float()

        graph_input = [x0]
        if self.use_pLDDT:
            graph_input.append(pLDDT)
        if self.use_bfactor:
            graph_input.append(bfactor)
        x0 = torch.cat(graph_input, dim=1)

        # pipe node features to linear layers
        fc_input = graph_input if self.feat2fc else []

        # pass each graph dimension through its own conv block
        for dim_idx, dim in enumerate(self.graph_dims):
            edge_type = ('residue', dim, 'residue')

            # if self.graph_dims == 'backbone':
            #     pass

            # drop edges
            dim_edge_index, _ = pyg.utils.dropout_edge(
                data_batch[edge_type].edge_index,
                p=self.dropedge_rate,
                force_undirected=True,
                training=self.training
            )

            # drop nodes
            dim_edge_index, _, node_mask = pyg.utils.dropout_node(
                dim_edge_index,
                p=self.dropnode_rate,
                num_nodes=data_batch['residue'].num_nodes,
                training=self.training
            )

            # keep features only retained nodes
            x0[~node_mask] = 0

            conv_block = self.conv_block_list[dim_idx]
            x = conv_block(x0, dim_edge_index, data_batch['residue'].batch)

            # pipe features from each graph dimension into the fc layer
            fc_input.append(x)

        if not fc_input and self.use_ogt:
            x = data_batch.ogt[:,None]

        elif fc_input:
            x = torch.cat(fc_input, dim=1)
            # unify dimensionality across proteins
            if self.global_pool == 'mean':
                x = pyg.nn.global_mean_pool(x, data_batch['residue'].batch)
            elif self.global_pool == 'max':
                x = pyg.nn.global_max_pool(x, data_batch['residue'].batch)
            elif self.global_pool == 'sum':
                x = pyg.nn.global_add_pool(x, data_batch['residue'].batch)
            else:
                raise ValueError('global_pool must be "mean" or "max"')

            if self.use_ogt:
                x = torch.cat((x, data_batch.ogt[:,None]), dim=1)

        for fc_block in self.fc_block_list:
            x = fc_block(x)

        return x

    def save_args(self, save_dir):
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

    def reset_parameters(self):

        # (re)initialize convolutional parameters
        for conv_block in self.conv_block_list:
            for layer in conv_block.children():
                if isinstance(layer, pyg.nn.conv.MessagePassing):
                    layer.reset_parameters()
                    # nn.init.kaiming_normal_(layer.lin.weight, a=0.01)
                    # nn.init.zeros_(layer.bias)
            # for name, param in mods.named_parameters():
            #     print(name, param.size())

        # (re)initialize fc parameters
        count = 1
        for fc_block in self.fc_block_list:
            for layer in fc_block.children():
                if isinstance(layer, nn.Linear):
                    if count < self.n_linear_layers:
                        nn.init.kaiming_normal_(layer.weight, a=0.01)
                        nn.init.zeros_(layer.bias)
                        count += 1
                    else:
                        nn.init.normal_(layer.weight)
                        nn.init.zeros_(layer.bias)
            # for name, param in self.fc_block.named_parameters():
            #     print(name, param.size())

    def eval(self):
        return self.train(False)

    def train(self, mode=True):

        if mode:
            self.training = True
        else:
            self.training = False

        # call parent function
        return super().train(mode)
