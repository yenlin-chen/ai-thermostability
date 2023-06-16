import os, sys, wandb, time
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
rand_gen = torch.Generator().manual_seed(69)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# for multiprocessing
if __name__== '__main__':

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
    # experiment setup
    ####################################################################
    # hyperparameters
    n_epochs = 300
    batch_size = 256
    learning_rate = 0.001
    split_ratio = [8,2]

    loss_type = 'mse'

    # file to save training history
    history_file = 'training_history.csv'
    prediction_file_valid = 'predicted_values-valid_set.csv'
    prediction_file_train = 'predicted_values-train_set.csv'

    # instantiate required objects for experiment
    dataset = DeepSTABp_Dataset(
        experiment='lysate',
        organism=None,
        cell_line=None,
        version='v2-pae',
        transform=norm_0to1
    )
    print()
    model = MultiGCN(
        graph_dims=['pae', 'contact', 'backbone', 'codir', 'codir', 'deform'],
        node_feat_name='x',
        dim_node_feat=1024,
        dim_node_hidden_ls=[64, 64, 64],
        dim_hidden_ls=[64],
        # dropout_rate=0.5,
        # dropedge_rate=0.2,
        dropfeat_rate=0.5,
        # dropnode_rate=0.3,
        # feat2fc=False,
        # conv_norm=True,
        # fc_norm=True,
        # global_pool='sum',
        # debug=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

    ### UPLOAD HYPERPARAMETERS AND METADATA TO WANDB
    wandb_section = ''
    wandb.init(
        project='thermostability-all_lysates',
        name=osp.basename(os.getcwd()),
        config=dict(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dataset=dataset.set_name,
            split_ratio=split_ratio,
            dataset_version=dataset.version,
            edge_types=model.all_args['graph_dims'],
            node_feat=model.all_args['node_feat_name'],
            node_hidden_size=model.all_args['dim_node_hidden_ls'],
            fc_hidden_size=model.all_args['dim_hidden_ls'],
            dropout_rate=model.all_args['dropout_rate'],
            dropedge_rate=model.all_args['dropedge_rate'],
            dropnode_rate=model.all_args['dropnode_rate'],
            dropfeat_rate=model.all_args['dropfeat_rate'],
            feat2fc=model.all_args['feat2fc'],
            conv_norm=model.all_args['conv_norm'],
            fc_norm=model.all_args['fc_norm'],
            global_pool=model.all_args['global_pool'],
            random_seed=rand_gen.initial_seed(),
            loss_fn=loss_type,
        ),
    )
    wandb.watch(model, log='gradients', log_freq=10)
    # configurate WandB
    wandb.define_metric(f'train_loss ({loss_type})', summary='min')
    wandb.define_metric(f'valid_loss ({loss_type})', summary='min')
    wandb.define_metric('train_mse', summary='min')
    wandb.define_metric('valid_mse', summary='min')
    wandb.define_metric('train_mae', summary='min')
    wandb.define_metric('valid_mae', summary='min')
    wandb.define_metric('train_rmse', summary='min')
    wandb.define_metric('valid_rmse', summary='min')
    wandb.define_metric('train_r2', summary='max')
    wandb.define_metric('valid_r2', summary='max')
    wandb.define_metric('train_pcc', summary='max')
    wandb.define_metric('valid_pcc', summary='max')

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

        ### LOG TO WANDB
        # performance
        item_name = [f'loss  ({loss_type})'] + list(metrics.keys())
        wandb.log({
            **{
                f'train_{item_name[i]}': t_metrics[i]
                for i in range(len(item_name))
            },
            **{
                f'valid_{item_name[i]}': v_metrics[i]
                for i in range(len(item_name))
            }
        })
        # predictions
        if epoch%25 == 0:# or epoch == 1:
            data = torch.dstack((t_outputs, t_labels)).squeeze().tolist()
            table = wandb.Table(data=data, columns=['predicted', 'true'])
            wandb.log({
                'scatter/train-true_vs_pred' : wandb.plot.scatter(
                    table, 'predicted', 'true', 'true vs pred (train)')
            })

            data = torch.dstack((v_outputs, v_labels)).squeeze().tolist()
            table = wandb.Table(data=data, columns=['predicted', 'true'])
            wandb.log({
                'scatter/valid-true_vs_pred' : wandb.plot.scatter(
                    table, 'predicted', 'true', 'true vs pred (valid)')
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
        f'train-true_vs_pred': wandb.Image('train-true_vs_pred.png'),
        f'valid-true_vs_pred': wandb.Image('valid-true_vs_pred.png'),
    })
