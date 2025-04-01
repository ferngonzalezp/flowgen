import lightning as L
from flowgen import tfno, hitDataModule
from lightning.pytorch.plugins.environments import MPIEnvironment, SLURMEnvironment
from argparse import ArgumentParser
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from flowgen.utils.loss import nrmse_loss, rH1loss
from lightning.pytorch.profilers import PyTorchProfiler
import os
import yaml


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
    torch.set_float32_matmul_precision('high')

    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    data_path = args.data_path
    val_dirs = [data_path+f"val/{case_name}" for case_name in args.cases]
    if args.ckpt_path:
        restart_reservoir = None
    else:
        restart_reservoir = True
    dm = hitDataModule(val_dirs=val_dirs, seq_len = args.seq_len, data_dir=args.stream_path, batch_size=args.batch_size, 
        reservoir_treshold=args.reservoir_size, 
        target_dims=[32,32,32], 
        restart_reservoir=restart_reservoir, 
        n_streams=args.n_streams,
        reservoir_per_node = args.reservoir_per_node)

    affine=False
    if args.use_affine:
        affine=True
    model = tfno(model_config=model_config,loss=args.loss, lr=args.lr, num_classes=len(val_dirs), affine=affine, model=args.model, 
            weight_decay=args.weight_decay, lr_warmup=args.lr_warmup, lr_warmup_steps=args.lr_warmup_steps)

    if args.save_path:
         
         
         if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        
         case_name_folder = '{}_{}_online'.format(args.model, args.loss)
         create_directory = True
         i = 1
         save_path = os.path.join(args.save_path, case_name_folder)
         while create_directory:
            if os.path.exists(save_path):
                case_name_folder_new = case_name_folder + "-%d" % i
                save_path = os.path.join(args.save_path, case_name_folder_new)
                i += 1
            else:
                create_directory   = False
        
    else:
        save_path = os.getcwd()
    
    profiler = PyTorchProfiler(filename="perf-logs", with_flops=True)
    
    trainer = L.Trainer(max_steps=args.steps, devices=args.devices, num_nodes=args.nodes, 
                        accelerator='auto',
                        plugins=SLURMEnvironment(),
                        max_epochs = None,
                        callbacks=[stopping_callback(), replace_criterion(), 
                                     checkpoint_callback, checkpoint_rst, val_avg_metric()],
                        val_check_interval=320, check_val_every_n_epoch=None,
                        log_every_n_steps=1,
                        strategy=DDPStrategy(find_unused_parameters=False),
                        precision=args.precision,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                        default_root_dir=save_path,
                        overfit_batches=args.overfit_batches,
                        limit_train_batches=args.limit_train_batches,
                        profiler='simple',
                        )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

def float_or_int(value):
    if "." in value:  # If it has a decimal point, treat it as a float
        try:
            float_val = float(value)
            if not (0.0 <= float_val <= 1.0):  # Ensure it's in the valid range
                raise argparse.ArgumentTypeError(f"Float value must be between 0.0 and 1.0, got {value}")
            return float_val
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float: {value}")
    else:  # Otherwise, treat it as an int
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int: {value}")

if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--loss", type=str)
     parser.add_argument("--model_config", type=str)
     parser.add_argument("--devices", type=int, default=1)
     parser.add_argument("--modes", type=int, default=16)
     parser.add_argument("--nodes", type=int, default=1)
     parser.add_argument("--steps", type=int, default=-1)
     parser.add_argument("--ckpt_path", type=str, default=None)
     parser.add_argument("--data_path", type=str)
     parser.add_argument("--stream_path", type=str)
     parser.add_argument("--batch_size", type=int, default=4)
     parser.add_argument("--lr", type=float, default=1e-3)
     parser.add_argument("--weight_decay", type=float, default=1e-3)
     parser.add_argument("--seq_len", type=int, nargs='+', default=[10, 100])
     parser.add_argument("--use_affine", action='store_true')
     parser.add_argument("--use_ema", action='store_true')
     parser.add_argument("--reservoir_per_node", action='store_true')
     parser.add_argument("--model", type=str, default='TFNO_t')
     parser.add_argument('--save_path', type=str, default=None)
     parser.add_argument('--overfit_batches', type=int, default=0)
     parser.add_argument('--reservoir_size', type=int, default=64)
     parser.add_argument('--limit_train_batches', type=float_or_int, default=1.0)
     parser.add_argument("--accumulate_grad_batches", type=int, default=1)
     parser.add_argument("--lr_warmup", action='store_true')
     parser.add_argument("--lr_warmup_steps", type=int, default=5)
     parser.add_argument("--precision", type=str, default="32")
     parser.add_argument("--cases", type=str, nargs='+', default=["case1", "case2", "case3"])
     parser.add_argument("--n_streams", type=int, default=3)
     args = parser.parse_args()
     main(args)