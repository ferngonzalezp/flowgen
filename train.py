import lightning as L
from flowgen import tfno, hitDataModule
from lightning.pytorch.plugins.environments import MPIEnvironment
from argparse import ArgumentParser
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from flowgen.utils.loss import nrmse_loss, rH1loss
import os


checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss_avg", mode='min', auto_insert_metric_name=True)
checkpoint_rst = ModelCheckpoint()

class val_avg_metric(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
            val_loss_avg =  [val_loss/n for val_loss, n in zip(pl_module.val_loss_avg, trainer.num_val_batches)]
            if pl_module.loss == 'dynamic':
                pl_module.dynamic_loss.update_metrics(val_loss_avg)
            pl_module.log('val_loss_avg', torch.mean(torch.tensor(val_loss_avg)).to(pl_module.device), sync_dist=True, prog_bar=True)
            pl_module.val_loss_avg = [0] * len(val_loss_avg)

class stopping_callback(L.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if pl_module.local_rank == 0:
                if trainer.datamodule.train_ds.end_of_stream == True:
                        return -1
            else:
                if trainer.datamodule.train_ds.end_of_stream == True:
                        return -1

class replace_criterion(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            y = batch[0]
            n = batch[1]
            time = batch[-1]
            y_pred = []
            input = y[...,0]
            crit = []
            with torch.no_grad():
                for i in range(y.shape[-1]-1):
                    pred = pl_module.predict(input, time[...,i], time[...,i+1])
                    y_pred.append(pred)
                    input = pred
                y_pred = torch.stack(y_pred, dim=-1)
                for i in range(y_pred.shape[0]):
                    crit.append(nrmse_loss(y_pred[i:i+1], y[i:i+1,...,1:]))
            
            for i in range(y_pred.shape[0]):
                trainer.datamodule.train_ds.reservoir.set_at_("loss", crit[i].cpu(), n[i])

            if batch_idx % 10 == 0:
                 if pl_module.local_rank == 0:
                    trainer.datamodule.train_ds.get_new_data()
            #print("Loss of batch: ", crit, " sample indexes: :", n)
            #print("loss per sample: ", trainer.datamodule.train_ds.reservoir["loss"])


def main(args):
    data_path = '/scratch/cfd/gonzalez/hit_online_training/04_HIT_decay/'
    train_data_dirs = [data_path+"train/case1", data_path+"train/case2", data_path+"train/case3"]
    val_dirs = [data_path+"case1", data_path+"case2", data_path+"case3"]
    dm = hitDataModule(val_dirs=val_dirs, seq_len =[10,100], data_dir="train_online", batch_size=args.batch_size, reservoir_treshold=200, target_dims=[32,32,32])
    model = tfno(loss=args.loss, lr=args.lr, precision='full', modes=16, num_classes=len(train_data_dirs), use_ema=args.use_ema, affine=False, model=args.model)

    if args.save_path:
         
         root_dir = args.save_path + '/{}_'.format(args.model)
         if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        
         create_directory = True
         i = 1
         while create_directory:
            if os.path.exists(root_dir):
                case_name_folder = "-%d" % i
                save_path = os.path.join(root_dir, case_name_folder)
                i += 1
            else:
                save_path     = os.path.join(root_dir, case_name_folder)
                create_directory   = False
        
    else:
        save_path = os.getcwd()
    
    
    trainer = L.Trainer(max_epochs=args.epochs, devices=args.devices, num_nodes=args.nodes, 
                        accelerator='gpu',
                        plugins=MPIEnvironment(),
                        callbacks = [stopping_callback(), replace_criterion(), 
                                     checkpoint_callback, checkpoint_rst, val_avg_metric()],
                        val_check_interval=500, check_val_every_n_epoch=None,
                        accumulate_grad_batches=8,
                        gradient_clip_val=5.0, gradient_clip_algorithm="norm",
                        strategy=DDPStrategy(find_unused_parameters=True),
                        default_root_dir=save_path
                        )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--loss", type=str)
     parser.add_argument("--devices", type=int, default=1)
     parser.add_argument("--modes", type=int, default=16)
     parser.add_argument("--nodes", type=int, default=1)
     parser.add_argument("--epochs", type=int, default=-1)
     parser.add_argument("--ckpt_path", type=str, default=None)
     parser.add_argument("--batch_size", type=int, default=4)
     parser.add_argument("--lr", type=float, default=1e-3)
     parser.add_argument("--seq_len", type=int, nargs='+', default=[10, 100])
     parser.add_argument("--use_affine", action='store_true')
     parser.add_argument("--use_ema", action='store_true')
     parser.add_argument("--model", type=str, default='TFNO_t')
     parser.add_argument('--save_path', type=str, default=None)
     args = parser.parse_args()
     main(args)