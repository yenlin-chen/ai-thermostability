if __name__ == '__main__':
    from check_model_args import check_args
else:
    from .check_model_args import check_args

import json
import sys
from os import path

import torch
from torch import nn
from torchinfo import summary

import torch_geometric as pyg

def simple_embedding_block(neuron_ls, dropout_rate,
                           activation='leakyrelu'):

        ## BUILD SEQUENTIAL MODEL
        mods = []
        for layer_idx in range(len(neuron_ls) - 1):
            dim_input = neuron_ls[layer_idx]
            dim_output = neuron_ls[layer_idx + 1]

            # linear connection
            mods.append(nn.Linear(dim_input, dim_output))

            # dropout
            mods.append(nn.Dropout(p=dropout_rate))

            # activation
            if activation == 'leakyrelu':
                mods.append(nn.LeakyReLU())
            elif activation == 'selu':
                mods.append(nn.SELU())

        return nn.Sequential(*mods)

class MultiGCN(nn.Module):
    def __init__(self,
                 # FEATURE SELECTION
                 graph_dims,
                 use_ogt=True,
                 feat2ffc=False,
                 use_node_pLDDT=False,
                 use_node_bfactor=False,
                 pLDDT2ffc=False,
                 bfactor2ffc=False,

                 # GRAPH CONVOLUTION SETUP
                 node_feat_name=None,
                 node_feat_size=None,
                 gnn_type=None,
                 gat_atten_heads=None,
                 dim_node_hidden_ls=None,
                 n_conv_layers=None,
                 dim_shape=None,
                 dim_node_hidden=None,
                 conv_norm=None,
                 norm_graph_input=None,
                 norm_graph_output=None,
                 graph_global_pool=None,
                 graph_dropout_rate=None,
                 dropfeat_rate=None,
                 dropedge_rate=None,
                 dropnode_rate=None,
                 jk_mode=None,

                 # GRAPH EMBEDDING SETUP
                 embed_graph_outputs=None,
                 graph_embedding_hidden_ls=None,
                 n_graph_embedding_layers=None,
                 graph_embedding_dim=None,
                 graph_embedding_dropout_rate=None,

                 # pLDDT EMBEDDING SETUP
                 embed_pLDDT=None,
                 pLDDT_dropout_rate=None,

                 # bfactor EMBEDDING SETUP
                 embed_bfactors=None,
                 bfactor_dropout_rate=None,

                 # OGT EMBEDDING SETUP
                 embed_ogt=None,
                 ogt_dropout_rate=None,

                 # FEAT2FFC SETUP
                 feat2ffc_feat_name=None,
                 feat2ffc_feat_size=None,
                 feat2ffc_global_pool=None,

                 # FEATURE REDUCTION SETUP
                 embed_feat2ffc=None,
                 feat2ffc_embedding_hidden_ls=None,
                 n_feat2ffc_embedding_layers=None,
                 feat2ffc_embedding_dim=None,
                 feat2ffc_dropout_rate=None,

                 # FC SETUP
                 fc_hidden_ls=None,
                 n_fc_hidden_layers=2,
                 fc_norm=True,
                 norm_fc_input=False,
                 fc_dropout_rate=0.5,

                 # OTHERS
                 sort_graph_dims=False,
                 debug=False):
        '''Instantiate all components with trainable parameters'''

        self.debug = debug
        if debug:
            torch.autograd.set_detect_anomaly(True)

        if sort_graph_dims:
            graph_dims.sort()

        ################################################################
        # CHECK CONSISTENCY OF ARGUMENTS
        ################################################################

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']
        check_args(**self.all_args)

        ################################################################
        # SAVE A COPY OF ARGUMENTS PASSED
        ################################################################

        # for cloning the model later
        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        # arguments required in forward()
        self.graph_dims = graph_dims
        self.feat2ffc = feat2ffc
        self.use_ogt = use_ogt
        self.dropedge_rate = dropedge_rate
        self.dropnode_rate = dropnode_rate
        self.use_node_pLDDT=use_node_pLDDT
        self.use_node_bfactor=use_node_bfactor
        self.pLDDT2ffc = pLDDT2ffc
        self.bfactor2ffc = bfactor2ffc
        self.node_feat_name = node_feat_name
        self.feat2ffc_feat_name = feat2ffc_feat_name

        # arguments required in reset_parameters()
        self.n_conv_layers = n_conv_layers
        self.graph_embedding_dim = graph_embedding_dim
        self.feat2ffc_embedding_dim = feat2ffc_embedding_dim
        self.embed_ogt = embed_ogt

        # internal parameters
        self.training = True
        self.debug = debug

        super().__init__()

        ################################################################
        # INSTANTIATE POOLING LAYERS
        ################################################################
        if graph_global_pool is None:
            self.graph_pool = None
        elif graph_global_pool == 'mean':
            self.graph_pool = pyg.nn.global_mean_pool
        elif graph_global_pool == 'max':
            self.graph_pool = pyg.nn.global_max_pool
        elif graph_global_pool == 'sum':
            self.graph_pool = pyg.nn.global_add_pool
        else:
            raise ValueError(
                f'`graph_global_pool` must be "mean", "max", or "sum", '
                f'not {graph_global_pool}'
            )

        if feat2ffc:
            if feat2ffc_global_pool == 'mean':
                self.feat_pool = pyg.nn.global_mean_pool
            elif feat2ffc_global_pool == 'max':
                self.feat_pool = pyg.nn.global_max_pool
            elif feat2ffc_global_pool == 'sum':
                self.feat_pool = pyg.nn.global_add_pool
            else:
                raise ValueError(
                    f'`feat2ffc_global_pool` must be "mean", "max", or "sum", '
                    f'not {feat2ffc_global_pool}'
                )

        ################################################################
        # INSTANTIATE PLDDT / B-FACTOR EMBEDDING LAYERS
        ################################################################

        # keep track of the number of feature for fc layers
        # sum all features sizes that will go into the final fc block
        dim_fc_input = 0

        # determine whether bfactor and pLDDT should be included
        n_add_feat = 0

        # keep track of the number of features for graph conv layers
        dim_graph_input = 0 if node_feat_size is None else node_feat_size

        ### pLDDT
        if embed_pLDDT:

            pLDDT_neuron_ls = [1, 20, 10]
            self.pLDDT_block = simple_embedding_block(
                pLDDT_neuron_ls, dropout_rate=pLDDT_dropout_rate
            )

            n_add_feat += 10 if use_node_pLDDT else 0
            dim_fc_input += 20 if pLDDT2ffc else 0

        elif embed_pLDDT == False:
            self.pLDDT_block = nn.Identity()

            n_add_feat += 1 if use_node_pLDDT else 0
            dim_fc_input += 2 if pLDDT2ffc else 0

        ### BFACTOR
        if embed_bfactors:

            bfactor_neuron_ls = [1, 20, 10]
            self.bfactor_block = simple_embedding_block(
                bfactor_neuron_ls, dropout_rate=bfactor_dropout_rate
            )

            n_add_feat += 10 if use_node_bfactor else 0
            dim_fc_input += 20 if bfactor2ffc else 0

        elif embed_bfactors == False:
            self.bfactor_block = nn.Identity()

            n_add_feat += 1 if use_node_bfactor else 0
            dim_fc_input += 2 if bfactor2ffc else 0

        dim_graph_input += n_add_feat

        ################################################################
        # INSTANTIATE CONVOLUTIONAL LAYERS
        ################################################################

        if dim_node_hidden_ls is None and graph_dims != []:
            if dim_shape == 'constant':
                dim_node_hidden_ls = [dim_node_hidden] * n_conv_layers
            elif dim_shape == 'linear':
                dim_sum = dim_graph_input + dim_node_hidden
                factor = dim_sum // n_conv_layers
                dim_node_hidden_ls = [
                    factor*i for i in range(1,n_conv_layers)[::-1]
                ] + [dim_node_hidden]
            elif dim_shape == 'exp':
                factor = (dim_node_hidden/dim_graph_input) ** (1/3)
                dim_node_hidden_ls = [
                    int(dim_graph_input * factor**i)
                    for i in range(1,n_conv_layers)
                ] + [dim_node_hidden]
            elif n_conv_layers == 1:
                dim_node_hidden_ls = [ dim_node_hidden ]
            else:
                raise ValueError(
                    f'`dim_shape` must be "constant", "linear", or "exp", '
                    f'not {dim_shape}'
                )

        if dim_node_hidden_ls is None:
            dim_node_hidden_ls = [] # NoneType cannot be concatenated
        dim_node_ls = [ dim_graph_input ] + dim_node_hidden_ls

        self.dim_node_ls = dim_node_ls

        # TODO: change ModuleList to ModuleDict
        self.conv_block_list = nn.ModuleList([]) if graph_dims != [] else None
        for _ in range(len(graph_dims)):

            mods = []

            for layer_idx in range(len(dim_node_hidden_ls)):
                dim_input = dim_node_ls[layer_idx]
                dim_output = dim_node_ls[layer_idx + 1]

                next_idx = layer_idx

                # only for the first layer
                if layer_idx == 0:
                    # dropfeat
                    if dropfeat_rate:
                        mods.append((
                            nn.Dropout(p=dropfeat_rate),
                            f'x{next_idx} -> x{layer_idx+1}'
                        ))
                        next_idx = layer_idx+1

                # exclude the first layer
                if layer_idx != 0 or norm_graph_input:
                    # normalization
                    if conv_norm:
                        mods.append((
                            pyg.nn.GraphNorm(dim_input),
                            f'x{next_idx}, batch -> x{layer_idx+1}'
                        ))
                        next_idx = layer_idx+1

                # convolution
                if gnn_type == 'gcn':
                    conv = pyg.nn.GCNConv(
                        dim_input, dim_output,
                        add_self_loops=True, bias=True
                    )
                elif gnn_type == 'gin':
                    intermediate = (dim_input+dim_output) // 2
                    gin_nn = nn.Sequential(
                        nn.Linear(dim_input, intermediate),
                        nn.BatchNorm1d(intermediate),
                        nn.ReLU(),
                        nn.Linear(intermediate, dim_output),
                        nn.ReLU()
                    )
                    conv = pyg.nn.GINConv(
                        nn=gin_nn,
                        train_eps=True
                    )
                elif gnn_type == 'gat':
                    assert dim_output % gat_atten_heads == 0
                    conv = pyg.nn.GATConv(
                        dim_input, dim_output//gat_atten_heads,
                        heads=gat_atten_heads, dropout=graph_dropout_rate,
                        add_self_loops=True
                    )
                else:
                    raise ValueError(
                        f'`gnn_type` must be "gcn", "gin", or "gat", '
                        f'not "{gnn_type}"'
                    )
                mods.append((
                    conv,
                    f'x{next_idx}, edge_index -> x{layer_idx+1}'
                ))

                # dropout
                if graph_dropout_rate:
                    mods.append((
                        nn.Dropout(p=graph_dropout_rate),
                        f'x{layer_idx+1} -> x{layer_idx+1}'
                    ))

                # activation
                mods.append((
                    nn.LeakyReLU(),
                    f'x{layer_idx+1} -> x{layer_idx+1}'
                ))

            feats = [f'x{i+1}' for i in range(len(dim_node_hidden_ls))]

            if jk_mode is not None:
                # jumping knowledge connections
                mods.append((lambda *x: [*x], ', '.join(feats)+' -> xs'))

                if jk_mode == 'lstm':
                    mods.append((
                            pyg.nn.JumpingKnowledge(
                                'lstm',
                                channels=dim_node_hidden,
                                num_layers=2
                            ), 'xs -> x'
                    ))
                    graph_embedding_size = dim_node_hidden

                elif jk_mode == 'cat':
                    mods.append((pyg.nn.JumpingKnowledge('cat'), 'xs -> x'))
                    graph_embedding_size = sum(dim_node_hidden_ls)

                elif jk_mode == 'max':
                    mods.append((pyg.nn.JumpingKnowledge('max'), 'xs -> x'))
                    graph_embedding_size = dim_node_hidden

                else:
                    raise ValueError(
                        f'`jk_mode` must be "lstm", "cat", or "max", '
                        f'not "{jk_mode}"'
                    )

                # normalization for node embeddings
                if norm_graph_output:
                    mods.append(
                        (pyg.nn.GraphNorm(graph_embedding_size),
                         'x, batch -> x')
                    )

                # sum size of embeddings accross all layers and graph dims
                total_node_dims = graph_embedding_size * len(graph_dims)

            else:
                # no jumping knowledge connections
                total_node_dims = dim_node_hidden * len(graph_dims)

                if norm_graph_output:
                    mods.append(
                        (pyg.nn.GraphNorm(dim_node_hidden),
                         f'x{feats[-1]}, batch -> x{feats[-1]}')
                    )

            self.conv_block_list.append(
                pyg.nn.Sequential('x0, edge_index, batch', mods)
            )

        ################################################################
        # INSTANTIATE GRAPH EMBEDDING LAYERS
        ################################################################

        if graph_embedding_dim is None:
            # features will be piped without embedding block

            if graph_dims != []:
                # no graph embedding block

                self.graph_embed_block = nn.Identity()

                # sum size of embeddings
                dim_fc_input += total_node_dims

        elif graph_embedding_dim == 0:
            raise NotImplementedError

        else:
            # features will be piped after going through embedding block

            if graph_embedding_hidden_ls is None:
                dim_sum = total_node_dims + graph_embedding_dim
                factor = dim_sum // n_graph_embedding_layers
                graph_embedding_hidden_ls = [
                    factor*i for i in range(1,n_graph_embedding_layers)[::-1]
                ]

            graph_embed_neuron_ls = [total_node_dims]
            graph_embed_neuron_ls += graph_embedding_hidden_ls
            graph_embed_neuron_ls.append(graph_embedding_dim)

            ## BUILD SEQUENTIAL MODEL
            graph_embed_block = []
            for layer_idx in range(len(graph_embed_neuron_ls)-1):
                dim_input = graph_embed_neuron_ls[layer_idx]
                dim_output = graph_embed_neuron_ls[layer_idx + 1]

                # linear connection
                graph_embed_block.append(nn.Linear(dim_input, dim_output))

                # dropout
                graph_embed_block.append(
                    nn.Dropout(p=graph_embedding_dropout_rate)
                )

                # activation
                graph_embed_block.append(nn.LeakyReLU())

            # normalization
            graph_embed_block.append(nn.BatchNorm1d(dim_output, affine=True))

            self.graph_embed_block = nn.Sequential(*graph_embed_block)

            # sum size of embeddings
            dim_fc_input += graph_embedding_dim

        ################################################################
        # INSTANTIATE OGT EMBEDDING LAYERS
        ################################################################

        if use_ogt:
            if embed_ogt:
                ogt_neuron_ls = [1,20,10]

                ## BUILD SEQUENTIAL MODEL
                ogt_block = []
                for layer_idx in range(len(ogt_neuron_ls) - 1):
                    dim_input = ogt_neuron_ls[layer_idx]
                    dim_output = ogt_neuron_ls[layer_idx + 1]

                    # linear connection
                    ogt_block.append(nn.Linear(dim_input, dim_output))

                    # dropout
                    ogt_block.append(nn.Dropout(p=ogt_dropout_rate))

                    # activation
                    ogt_block.append(nn.LeakyReLU())

                self.ogt_block = nn.Sequential(*ogt_block)

                # sum size of embeddings
                dim_fc_input += ogt_neuron_ls[-1]

            else:
                self.ogt_block = nn.Identity()
                # sum size of embeddings
                dim_fc_input += 1

        ################################################################
        # INSTANTIATE FEAT2FFC EMBEDDING LAYERS
        ################################################################

        if feat2ffc:
            if embed_feat2ffc:
                # features will be piped after dimensionality reduction

                if feat2ffc_embedding_hidden_ls is None:
                    dim_sum = feat2ffc_feat_size + feat2ffc_embedding_dim
                    factor = dim_sum // n_feat2ffc_embedding_layers
                    feat2ffc_embedding_hidden_ls = [
                        factor*i for i in range(1,n_feat2ffc_embedding_layers)[::-1]
                    ]

                feat_reduce_neuron_ls = [feat2ffc_feat_size]
                feat_reduce_neuron_ls += feat2ffc_embedding_hidden_ls
                feat_reduce_neuron_ls.append(feat2ffc_embedding_dim)

                ## BUILD SEQUENTIAL MODEL
                feat_reduce_block = []

                for layer_idx in range(len(feat_reduce_neuron_ls)-1):
                    dim_input = feat_reduce_neuron_ls[layer_idx]
                    dim_output = feat_reduce_neuron_ls[layer_idx + 1]

                    # linear connection
                    feat_reduce_block.append(nn.Linear(dim_input, dim_output))

                    # dropout
                    feat_reduce_block.append(nn.Dropout(p=feat2ffc_dropout_rate))

                    # activation
                    feat_reduce_block.append(nn.LeakyReLU())

                # normalization
                feat_reduce_block.append(nn.BatchNorm1d(dim_output, affine=True))

                self.feat_reduce_block = nn.Sequential(*feat_reduce_block)

                # sum size of embeddings
                dim_fc_input += feat2ffc_embedding_dim

            else:
                # features will be piped without dimensionality reduction

                self.feat_reduce_block = nn.Identity()

                # sum size of embeddings
                dim_fc_input += feat2ffc_feat_size

        ################################################################
        # INSTANTIATE FULLY CONNECTED LAYERS
        ################################################################

        if fc_hidden_ls is None:
            factor = dim_fc_input//(n_fc_hidden_layers+1)
            if factor !=0:
                fc_hidden_ls = [
                    factor*i for i in range(1,n_fc_hidden_layers+1)[::-1]
                ]
            else:
                fc_hidden_ls = [1] * n_fc_hidden_layers

        fc_neuron_ls = [dim_fc_input] + fc_hidden_ls + [1]

        self.n_linear_layers = len(fc_neuron_ls) - 1

        fc_block = []
        for layer_idx in range(self.n_linear_layers):
            dim_input = fc_neuron_ls[layer_idx]
            dim_output = fc_neuron_ls[layer_idx + 1]

            # normalization
            if fc_norm and (layer_idx != 0 or norm_fc_input):
                fc_block.append(nn.BatchNorm1d(dim_input, affine=True))

            # linear connection
            fc_block.append(nn.Linear(dim_input, dim_output))

            # for non-output layers
            if layer_idx != (self.n_linear_layers-1):

                # dropout
                if fc_dropout_rate:
                    fc_block.append(nn.Dropout(p=fc_dropout_rate))

                # activation
                fc_block.append(nn.LeakyReLU())

        self.fc_block = nn.Sequential(*fc_block)

    def forward(self, data_batch):
        '''Make connects between the components to complete the model'''

        ################################################################
        # GRAPH INPUT PREPARATION
        ################################################################

        # get node features
        bfactor = data_batch['residue'].bfactor.float()[:, None]
        pLDDT = data_batch['residue'].pLDDT.float()[:, None]
        ogt = data_batch.ogt.float()[:,None]
        # res1hot = data_batch['residue'].res1hot.float()

        # batch metadata
        batch_vector = data_batch['residue'].batch.long()

        # gather all inputs for GNN
        if self.graph_dims != []:
            node_feat = getattr(data_batch['residue'], self.node_feat_name).float()
            graph_input = [node_feat]

        # feat2ffc
        if self.feat2ffc:
            feat2ffc_feat = getattr(
                data_batch['residue'], self.feat2ffc_feat_name
            ).float()

        # pipe node features to linear layers
        node_embeddings = []

        ################################################################
        # pLDDT EMBEDDING
        ################################################################

        pLDDT_graph_level_embedding = None

        if self.use_node_pLDDT or self.pLDDT2ffc:
            pLDDT_embedding = self.pLDDT_block(pLDDT)

            if self.use_node_pLDDT:
                graph_input.append(pLDDT_embedding)

            if self.pLDDT2ffc:
                pLDDT_graph_level_embedding = torch.cat([
                    pyg.nn.global_mean_pool(pLDDT_embedding, batch_vector),
                    pyg.nn.global_max_pool(pLDDT_embedding, batch_vector)
                ], dim=1)

        ################################################################
        # bfactor EMBEDDING
        ################################################################

        bfactor_graph_level_embedding = None

        if self.use_node_bfactor or self.bfactor2ffc:
            bfactor_embedding = self.bfactor_block(bfactor)

            if self.use_node_bfactor:
                graph_input.append(bfactor_embedding)

            if self.bfactor2ffc:
                bfactor_graph_level_embedding = torch.cat([
                    pyg.nn.global_mean_pool(bfactor_embedding, batch_vector),
                    pyg.nn.global_max_pool(bfactor_embedding, batch_vector)
                ], dim=1)

        ################################################################
        # GRAPH CONVOLUTIONS
        ################################################################

        if self.graph_dims != []:
            graph_input = torch.cat(graph_input, dim=1)

        # pass each graph dimension through its own conv block
        for dim_idx, dim in enumerate(self.graph_dims):
            edge_type = ('residue', dim, 'residue')

            # if self.graph_dims == 'backbone':
            #     pass

            dim_edge_index = data_batch[edge_type].edge_index.long()

            # drop edges
            if self.dropedge_rate:
                dim_edge_index, _ = pyg.utils.dropout_edge(
                    dim_edge_index,
                    p=self.dropedge_rate,
                    force_undirected=True,
                    training=self.training
                )

            # drop nodes
            if self.dropnode_rate:
                dim_edge_index, _, node_mask = pyg.utils.dropout_node(
                    dim_edge_index,
                    p=self.dropnode_rate,
                    num_nodes=data_batch['residue'].num_nodes,
                    training=self.training
                )

                # keep features only for retained nodes
                graph_input = graph_input * node_mask[:, None]

                # update batch vector to match new number of nodes
                batch_vector = batch_vector[node_mask]

            # pipe features from each graph dimension into the fc layer
            node_embeddings.append(
                self.conv_block_list[dim_idx](
                    graph_input,
                    dim_edge_index,
                    batch_vector
                )
            )

        if self.graph_dims != []:
            node_embeddings = torch.cat(node_embeddings, dim=1)
            graph_embedding = self.graph_pool(
                node_embeddings, batch_vector
            )
            # pass extracted features through embedding block
            graph_embedding = self.graph_embed_block(graph_embedding)
        else:
            graph_embedding = None

        ################################################################
        # OGT EMBEDDING
        ################################################################

        if self.use_ogt:
            ogt_embedding = self.ogt_block(ogt)
        else:
            ogt_embedding = None

        ################################################################
        # FEATURE EMBEDDING
        ################################################################

        # reduce dimensionality of node features (ProteinBERT / residue OHE)
        if self.feat2ffc:
            feat_embedding = self.feat_pool(feat2ffc_feat, batch_vector)
            feat_embedding = self.feat_reduce_block(feat_embedding)
        else:
            feat_embedding = None

        ################################################################
        # FC INPUT PREPARATION
        ################################################################

        # concatenate embeddings
        fc_input = torch.cat(
            [
                e for e in [
                    graph_embedding,
                    ogt_embedding,
                    feat_embedding,
                    pLDDT_graph_level_embedding,
                    bfactor_graph_level_embedding
                ]
                if e is not None
            ], dim=1
        )

        ################################################################
        # FC LAYERS
        ################################################################
        x = self.fc_block(fc_input)

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

        if self.n_conv_layers:
            # (re)initialize convolutional parameters
            for conv_block in self.conv_block_list:
                for layer in conv_block.children():
                    if isinstance(layer, pyg.nn.conv.MessagePassing):
                        layer.reset_parameters()
                        # nn.init.kaiming_normal_(layer.lin.weight, a=0.01)
                        # nn.init.zeros_(layer.bias)
                # for name, param in mods.named_parameters():
                #     print(name, param.size())

        if self.graph_embedding_dim:
            # (re)initialize graph embedding parameters
            for layer in self.graph_embed_block.children():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                if isinstance(layer, nn.BatchNorm1d):
                    layer.reset_parameters()

        if self.feat2ffc_embedding_dim:
            # (re)initialize feature reducing parameters
            for layer in self.feat_reduce_block.children():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                if isinstance(layer, nn.BatchNorm1d):
                    layer.reset_parameters()

        if self.embed_ogt:
            # (re)initialize ogt parameters
            for layer in self.ogt_block.children():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                if isinstance(layer, nn.BatchNorm1d):
                    layer.reset_parameters()

        # (re)initialize fc parameters
        count = 1
        for layer in self.fc_block.children():
            if isinstance(layer, nn.Linear):
                if count < self.n_linear_layers:
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                    count += 1
                else:
                    nn.init.normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()

    def eval(self):
        return self.train(False)

    def train(self, mode=True):

        if mode:
            self.training = True
        else:
            self.training = False

        # call parent function
        return super().train(mode)

class DeepSTABp(nn.Module):
    def __init__(self, use_ogt=True, debug=False):
        '''Instantiate all components with trainable parameters'''

        self.debug = debug
        if debug:
            torch.autograd.set_detect_anomaly(True)

        self.use_ogt = use_ogt

        super().__init__()

        ################################################################
        # SAVE A COPY OF ARGUMENTS PASSED
        ################################################################

        # for cloning the model later
        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        ################################################################
        # INSTANTIATE POOLING LAYER
        ################################################################

        self.feat_pool = pyg.nn.global_mean_pool
        dim_fc_input = 1024

        ################################################################
        # INSTANTIATE EXPERIMENT-TYPE BLOCK
        ################################################################

        fc_neuron_ls = [1,20,10]

        self.lysate_block = simple_embedding_block(
            fc_neuron_ls, dropout_rate=0.2,
            activation='selu'
        )
        dim_fc_input += 10

        self.cell_block = simple_embedding_block(
            fc_neuron_ls, dropout_rate=0.2,
            activation='selu'
        )
        dim_fc_input += 10

        ################################################################
        # INSTANTIATE OGT BLOCK
        ################################################################

        if use_ogt:
            fc_neuron_ls = [1,20,10]

            self.ogt_block = simple_embedding_block(
                fc_neuron_ls, dropout_rate=0.2,
                activation='selu'
            )

            dim_fc_input += 10

        ################################################################
        # INSTANTIATE FULLY CONNECTED LAYERS
        ################################################################

        fc_neuron_ls = [dim_fc_input, 4098, 512, 256, 128, 1]

        fc_block = []
        for layer_idx in range(len(fc_neuron_ls) - 1):
            dim_input = fc_neuron_ls[layer_idx]
            dim_output = fc_neuron_ls[layer_idx + 1]

            # linear connection
            fc_block.append(nn.Linear(dim_input, dim_output))

            if layer_idx != len(fc_neuron_ls) - 2:
                # activation
                fc_block.append(nn.SELU())

                # normalization
                fc_block.append(nn.BatchNorm1d(dim_output, affine=True))

                # dropout
                fc_block.append(nn.Dropout(p=0.2))

        self.fc_block = nn.Sequential(*fc_block)

    def forward(self, data_batch):
        '''Make connects between the components to complete the model'''

        ################################################################
        # INPUT PREPARATION
        ################################################################
        ogt = data_batch.ogt[:,None].float()
        node_feat = data_batch['residue'].x.float()
        batch_vector = data_batch['residue'].batch.long()

        cell = torch.zeros_like(ogt)
        lysate = torch.ones_like(ogt)

        cell_embedding = self.cell_block(cell)
        lysate_embedding = self.lysate_block(lysate)

        ################################################################
        # FC INPUT PREPARATION
        ################################################################

        feat_embedding = self.feat_pool(node_feat, batch_vector)

        if self.use_ogt:
            ogt_embedding = self.ogt_block(ogt)
            fc_input = torch.cat([
                cell_embedding,
                lysate_embedding,
                feat_embedding,
                ogt_embedding,
            ], dim=1)
        else:
            fc_input = torch.cat([
                cell_embedding,
                lysate_embedding,
                feat_embedding,
            ], dim=1)

        ################################################################
        # FC LAYERS
        ################################################################
        x = self.fc_block(fc_input)

        return x

    def save_args(self, save_dir):

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

    def reset_parameters(self):

        # (re)initialize ogt parameters
        if self.use_ogt:
            for layer in self.ogt_block.children():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, nonlinearity='linear'
                    )
                    nn.init.zeros_(layer.bias)
                if isinstance(layer, nn.BatchNorm1d):
                    layer.reset_parameters()

        # (re)initialize fc parameters
        count = 1
        for layer in self.fc_block.children():
            if isinstance(layer, nn.Linear):
                if count < 5:
                    nn.init.kaiming_normal_(
                        layer.weight, nonlinearity='linear'
                    )
                    nn.init.zeros_(layer.bias)
                    count += 1
                else:
                    nn.init.normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()

    def eval(self):
        return self.train(False)

    def train(self, mode=True):

        if mode:
            self.training = True
        else:
            self.training = False

        # call parent function
        return super().train(mode)
