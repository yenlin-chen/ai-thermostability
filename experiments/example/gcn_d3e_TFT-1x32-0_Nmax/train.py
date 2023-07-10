import os, sys, wandb, time, yaml
import os.path as osp
self_dir = osp.normpath(os.getcwd())
root_dir = osp.dirname(osp.dirname(osp.dirname(self_dir)))
package_dir = osp.normpath(osp.join(root_dir, 'src'))
sys.path.append(package_dir)

from ml_modules.training.model_arch import MultiGCN
from ml_modules.training.trainer import Trainer
from ml_modules.training.metrics import pcc, rmse, mae, mse, r2
from ml_modules.data.datasets import DeepSTABp_Dataset
from ml_modules.data.transforms import norm_0to1

import random, torch, torchinfo
import numpy as np

import torch_geometric as pyg
# from tqdm import tqdm
import matplotlib.pyplot as plt

# fix random generator for reproducibility
random_seed = 69
rand_gen = torch.Generator().manual_seed(random_seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# for multiprocessing
def main():

    metrics = {
        'pcc': pcc,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'r2': r2
    }

    # machine-specific parameters
    num_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {num_workers} workers for dataloading')
    print(f'Training on {device} device\n')

    ####################################################################
    # initialize wandb
    ####################################################################
    run = wandb.init(
        project='thermostability-dim_reduced',
        name=osp.basename(os.getcwd())
    )

    # dim_node = 2 ** wandb.config.dim_node_power
    # dim_node_hidden_ls = [dim_node] * wandb.config.n_gcn_layers

    # dim_graph_embedding = sum(dim_node_hidden_ls) * len(wandb.config.edge_types)
    # if wandb.config.feat2fc:
    #     dim_fc_input_approx = int(dim_graph_embedding + 1024)
    # else:
    #     dim_fc_input_approx = int(dim_graph_embedding)
    # fc_hidden_ls = np.linspace(
    #     dim_fc_input_approx, 1, wandb.config.n_fc_layers+2
    # )[1:-1].astype(np.int_).tolist()

    ####################################################################
    # experiment setup
    ####################################################################

    # hyperparameters
    n_epochs = 150
    batch_size = 64 # 2 ** wandb.config.batch_size_power
    learning_rate = 0.01 # 2 ** wandb.config.lr_power
    split_ratio = [8,2]

    loss_type = 'mse'

    # instantiate required objects for experiment
    dataset = DeepSTABp_Dataset(
        experiment='lysate',
        organism=None,
        cell_line=None,
        version='v3-ogt',
        transform=norm_0to1
    )
    print()
    model = MultiGCN(
        # FEATURE SELECTION
        graph_dims=['pae', 'contact', 'backbone', 'codir', 'coord', 'deform'],
        node_feat_name='x',
        dim_node_feat=1024,
        use_ogt=True,
        feat2fc=False,
        use_node_pLDDT=True,
        use_node_bfactor=True,
        use_fc_pLDDT=False,
        use_fc_bfactor=False,

        # GRAPH CONVOLUTION SETUP
        gnn_type='gcn',
        gat_atten_heads=None,
        dim_node_hidden_ls=None,
        n_conv_layers=1,
        dim_shape='constant',
        dim_node_hidden=32,
        conv_norm=True,
        norm_graph_input=True,
        norm_graph_output=False,
        graph_global_pool='max',
        graph_dropout_rate=0,
        dropfeat_rate=0,
        dropedge_rate=0,
        dropnode_rate=0,
        jk_mode='cat',

        # GRAPH EMBEDDING SETUP
        use_graph_embedding=False,
        graph_embedding_hidden_ls=None,
        n_graph_embedding_layers=None,
        graph_embedding_dim=None,
        graph_embedding_dropout_rate=None,

        # pLDDT EMBEDDING SETUP
        use_pLDDT_embedding=False,
        pLDDT_dropout_rate=None,

        # bfactor EMBEDDING SETUP
        use_bfactor_embedding=False,
        bfactor_dropout_rate=None,

        # OGT EMBEDDING SETUP
        use_ogt_embedding=True,
        ogt_dropout_rate=0.2,

        # FEATURE REDUCTION SETUP
        use_feat_reduce=None,
        feat_reduce_hidden_ls=None,
        n_feat_reduce_layers=None,
        feat_reduce_dim=None,
        feat_global_pool=None,
        feat_reduce_dropout_rate=None,

        # FC SETUP
        fc_hidden_ls=None,
        n_fc_hidden_layers=2,
        fc_norm=True,
        norm_fc_input=True,
        fc_dropout_rate=0.5,

        # OTHERS
        debug=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, amsgrad=False
    )
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular',
    #     cycle_momentum=False
    # )
    if loss_type == 'mae':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    elif loss_type == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif loss_type == 'smae':
        loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    else:
        raise ValueError

    # how many parameters?
    print()
    print(model)
    print()
    torchinfo.summary(model)
    # save a copy for reference
    model.save_args('.')
    wandb.save('model-summary.txt')

    ## UPLOAD HYPERPARAMETERS AND METADATA TO WANDB
    wandb.config.update({
        **{
            # training setup
            'random_seed': rand_gen.initial_seed(),
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'loss_fn': loss_type,
            # dataset-related
            'dataset': dataset.set_name,
            'dataset_version': dataset.version,
            'split_ratio': split_ratio,
        }, **model.all_args
    })
    run.log_code()
    wandb.watch(model, log='gradients', log_freq=1000)
    # configurate WandB
    wandb.define_metric(
        f'train.loss_{loss_type}', summary='min', goal='minimize',
        step_metric='epoch', hidden=True
    )
    for p in ['valid', 'train']:
        for m in ['mse', 'mae', 'rmse']:
            wandb.define_metric(
                f'{p}.{m}', summary='min',
                goal='minimize', step_metric='epoch'
            )
        for m in ['pcc', 'r2']:
            wandb.define_metric(
                f'{p}.{m}', summary='max',
                goal='maximize', step_metric='epoch'
            )

    # file to save training history
    history_file = 'training_history.csv'
    prediction_file_valid = 'predicted_values-valid_set.csv'
    prediction_file_train = 'predicted_values-train_set.csv'

    ####################################################################
    # split dataset into train and valid
    ####################################################################
    split_total = np.sum(split_ratio)
    n_train_batches = len(dataset) * split_ratio[0]/split_total // 512
    n_train_data = int(n_train_batches * 512)
    n_valid_data = len(dataset) - n_train_data

    train_set, valid_set = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[n_train_data, n_valid_data],
        generator=rand_gen
    )
    # train_set = valid_set # debug

    ### INSTANTIATE DATALOADERS
    train_loader = pyg.loader.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=rand_gen
    )
    train_size = len(train_set)
    train_accessions_ordered = dataset.processable_accessions[train_set.indices]
    train_order = { train_accessions_ordered[i]: i for i in range(train_size) }

    valid_loader = pyg.loader.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=rand_gen
    )
    valid_size = len(valid_set)
    valid_accessions_ordered = dataset.processable_accessions[valid_set.indices]
    valid_order = {valid_accessions_ordered[i]: i for i in range(valid_size)}

    ### EXPORT ENTRY IDENTIFIERS FOR FUTURE REFERENCE
    np.savetxt('training_entries.txt',
               train_accessions_ordered,
               fmt='%s')
    np.savetxt('validation_entries.txt',
               valid_accessions_ordered,
               fmt='%s')

    ### PLOT TRAIN / VALID DISTRIBUTION
    train_Tm = [dataset.Tm_dict[a] for a in train_accessions_ordered]
    valid_Tm = [dataset.Tm_dict[a] for a in valid_accessions_ordered]

    title = f'{dataset.organism}'
    title += f'-{dataset.experiment}' if dataset.experiment else ''
    title += f'_{dataset.cell_line}' if dataset.cell_line else ''
    plt.figure('hist')
    plt.hist(train_Tm, density=True, label=f'train ({train_size} entries)',
             alpha=0.7)
    plt.hist(valid_Tm, density=True, label=f'valid ({valid_size} entries)',
             alpha=0.7)
    plt.title(title)
    plt.xlabel('Tm (°C)')
    plt.ylabel('density')
    plt.legend()
    plt.savefig('data_split-Tm_distr.png', dpi=300, bbox_inches='tight')
    plt.close()

    ### DATASET STATISTICS
    train_min,  train_max = np.amin(train_Tm), np.amax(train_Tm)
    train_mean, train_std = np.mean(train_Tm), np.std(train_Tm)
    train_median = np.median(train_Tm)

    valid_min,  valid_max = np.amin(valid_Tm), np.amax(valid_Tm)
    valid_mean, valid_std = np.mean(valid_Tm), np.std(valid_Tm)
    valid_median = np.median(valid_Tm)

    # save to file
    lines = [
        '# dataset min max mean std median',
        f'train {train_min} {train_max} {train_mean} {train_std} {train_median}',
        f'valid {valid_min} {valid_max} {valid_mean} {valid_std} {valid_median}',
    ]
    with open('dataset_statistics.csv', 'w+') as f:
        f.write('\n'.join(lines)+'\n')

    ####################################################################
    # train / valid loop
    ####################################################################

    ### INSTANTIATE THE MODEL-TRAINING CONTROLLER
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_workers=num_workers,
        device=device,
        # CAUTION: RISK OF DATA LEAKAGE
        min_max=(np.amin(train_Tm), np.amax(train_Tm)),
        mean_std=(np.mean(train_Tm), np.std(train_Tm)),
    )

    ### FILE TO KEEP TRACK OF TRAINING PERFORMANCE
    # training history
    header = '# epoch,train_loss,valid_loss'
    for m in metrics.keys():
        header += f',train_{m},valid_{m}'
    with open(history_file, 'w+') as f:
        f.write(header + '\n')

    # prediction on validation set
    header = '# epoch,' + ','.join(valid_accessions_ordered)
    line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in valid_Tm])
    with open(prediction_file_valid, 'w+') as f:
        f.write(header +'\n')
        f.write(line_1 +'\n')
    # prediction on training set
    header = '# epoch,' + ','.join(train_accessions_ordered)
    line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in train_Tm])
    with open(prediction_file_train, 'w+') as f:
        f.write(header +'\n')
        f.write(line_1 +'\n')

    ### TRAIN FOR N_EPOCHS
    # pbar = tqdm(range(n_epochs), dynamic_ncols=True, ascii=True)
    for i in range(n_epochs):
        epoch = i+1
        print(f'Epoch {epoch}')

        # get leraning rate of this epoch
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        # time it
        start = time.time()

        ### ONE PASS OVER TRAINING SET
        # pbar.set_description(f'Ep{epoch:3d} (Train Pass)')
        t_loss, t_outputs, t_labels, t_accessions = trainer.train_one_epoch(
            train_loader
        )
        print(f'    train loss: {t_loss:.8f}')

        # compute various metrics
        t_metrics = [t_loss] + [
            m(t_outputs, t_labels) for m in metrics.values()
        ]

        ### ONE PASS OVER VALID SET
        # pbar.set_description(f'Ep{epoch:3d} (Valid Pass)')
        v_loss, v_outputs, v_labels, v_accessions = trainer.evaluate(
            valid_loader
        )
        print(f'    valid loss: {v_loss:.8f}')

        # compute various metrics
        v_metrics = [v_loss] + [
            m(v_outputs, v_labels) for m in metrics.values()
        ]

        ### SAVE MODEL PERFORMANCE
        line = f'{epoch}'
        for i in range(len(metrics)+1): # metrics + loss
            line += f',{t_metrics[i]:.8f},{v_metrics[i]:.8f}'
        with open(history_file, 'a+') as f:
            f.write(line + '\n')

        ### SAVE PREDICTION
        # VALIDATION SET
        # order outputted values by acccession
        idx_order = np.argsort(
            [valid_order[a] for a in v_accessions.tolist()]
        )
        v_outputs_ordered = v_outputs.detach().cpu().numpy()[idx_order]
        line = f'{epoch},' + ','.join(v_outputs_ordered.astype(np.str_))
        with open(prediction_file_valid, 'a+') as f:
            f.write(line + '\n')

        ### SAVE PREDICTION FOR TRAINING SET
        # order outputted values by acccession
        idx_order = np.argsort(
            [train_order[a] for a in t_accessions.tolist()]
        )
        t_outputs_ordered = t_outputs.detach().cpu().numpy()[idx_order]
        line = f'{epoch},' + ','.join(t_outputs_ordered.astype(np.str_))
        with open(prediction_file_train, 'a+') as f:
            f.write(line + '\n')

        ### SAVE MODEL PARAMETERS PERODICALLY
        if epoch%100 == 0:
            torch.save(model.state_dict(), f'model-ep{epoch}.pt')
            # wandb.save(f'model-ep{epoch}.pt')

        ### LOG TO WANDB
        # performance
        item_name = [f'loss_{loss_type}'] + list(metrics.keys())
        wandb.log({
            'epoch': epoch,
            'train': {
                item_name[i]: t_metrics[i] for i in range(len(item_name))
            },
            'valid': {
                item_name[i]: v_metrics[i] for i in range(len(item_name))
            },
            'lr': current_lr
        })

        # time it
        print(f' >> Time Elapsed: {time.time()-start:.4f}s\n')

    ####################################################################
    # save model parameters
    ####################################################################
    if epoch%100 != 0:
        torch.save(model.state_dict(), f'model-ep{epoch}.pt')

    ####################################################################
    # plot true vs pred on last epoch
    ####################################################################

    plt.scatter(
        t_labels.detach().cpu().numpy(),
        t_outputs.detach().cpu().numpy(),
        marker='x', s=1, alpha=0.7, zorder=3
    )
    plt.plot(np.linspace(30,95), np.linspace(30,95),
        '--', c='k', alpha=0.3, zorder=1)
    plt.title(f'mae: {t_metrics[2]:.2f}, rmse: {t_metrics[1]:.2f}, \n'
              f'r2: {t_metrics[4]:.2f}, pcc: {t_metrics[0]:.2f}')
    plt.xlabel('true Tm (Celcius)')
    plt.ylabel('predicted Tm (Celcius)')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.savefig('train-true_vs_pred.png', dpi=300, bbox_inches='tight')
    # wandb.log({
    #     'true vs pred (train)': plt
    # })
    plt.close()

    plt.scatter(
        v_labels.detach().cpu().numpy(),
        v_outputs.detach().cpu().numpy(),
        marker='x', s=1, alpha=0.7, zorder=3
    )
    plt.plot(np.linspace(30,95), np.linspace(30,95),
        '--', c='k', alpha=0.3, zorder=1)
    plt.title(f'mae: {v_metrics[3]:.2f}, rmse: {v_metrics[2]:.2f}, \n'
              f'r2: {v_metrics[5]:.2f}, pcc: {v_metrics[1]:.2f}')
    plt.xlabel('true Tm (Celcius)')
    plt.ylabel('predicted Tm (Celcius)')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.savefig('valid-true_vs_pred.png', dpi=300, bbox_inches='tight')
    # wandb.log({
    #     'true vs pred (valid)': plt
    # })
    plt.close()

    wandb.log({
        'train': { 'true_vs_pred': wandb.Image('train-true_vs_pred.png') },
        'valid': { 'true_vs_pred': wandb.Image('valid-true_vs_pred.png') }
    })

if __name__ == '__main__':
    main()
