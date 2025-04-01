import lightning as L
from flowgen import tfno, hitOfflineDataModule
from lightning.pytorch.plugins.environments import MPIEnvironment
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from argparse import ArgumentParser
import torch
from flowgen.utils.postprocess import Postprocess
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.distributed.fsdp import MixedPrecision
from mpi4py import MPI
import yaml
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
            
def main(args):
    torch.set_float32_matmul_precision('high')
    data_path = args.data_path #Path to dataset location

    train_data_dirs = [data_path+f"train/{case_name}" for case_name in args.cases]
    val_dir = [data_path+f"val/{case_name}" for case_name in args.cases]
    dm = hitOfflineDataModule(val_dirs = val_dir, data_dir = train_data_dirs, batch_size =  args.batch_size, seq_len=args.seq_len)
    affine=False
    if args.use_affine:
        affine=True

    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    model = tfno(model_config=model_config, loss=args.loss, lr=args.lr, num_classes=len(train_data_dirs), affine=affine, model=args.model, 
            weight_decay=args.weight_decay, lr_warmup=args.lr_warmup, lr_warmup_steps=args.lr_warmup_steps)

    fsdp_strategy = FSDPStrategy(
        # Default: Shard weights, gradients, optimizer state (1 + 2 + 3)
        #sharding_strategy="FULL_SHARD",
        # Shard gradients, optimizer state (2 + 3)
        sharding_strategy="SHARD_GRAD_OP",
        # Full-shard within a machine, replicate across machines
        #sharding_strategy="HYBRID_SHARD",
        # Don't shard anything (similar to DDP)
        #sharding_strategy="NO_SHARD",
    )

    if args.save_path:
         
         
         if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        
         case_name_folder = '{}_{}'.format(args.model, args.loss)
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
        

    trainer = L.Trainer(max_epochs=args.epochs, devices=args.devices, num_nodes=args.nodes, 
                        accelerator='gpu',
                        plugins=MPIEnvironment(),
                        callbacks=[checkpoint_callback, checkpoint_rst, val_avg_metric()],
                        strategy=DDPStrategy(find_unused_parameters=True),
                        precision=args.precision,
                        #detect_anomaly=False,
                        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        default_root_dir=save_path,
                        overfit_batches=args.overfit_batches,
                        limit_train_batches=args.limit_train_batches,
                        profiler='simple'
                        )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank==0:
        device = torch.device('cuda')
        best_model_pth = checkpoint_callback.best_model_path
        model = tfno.load_from_checkpoint(best_model_pth, loss=args.loss, lr=args.lr, precision='full', modes=16, num_classes=len(train_data_dirs), model=args.model,
                                          lr_warmup=args.lr_warmup, lr_warmup_steps=args.lr_warmup_steps, model_config=model_config)
        
        if '16' in args.precision:
            dtype = 'half'
        else:
            dtype = None
        dm = hitOfflineDataModule(val_dirs = val_dir, data_dir = train_data_dirs, batch_size =  args.batch_size, seq_len=[2,100])
        dm.setup('fit')
        postpro = Postprocess(dm.val_dataloader(), model, trainer.log_dir+'/results', device, val_dir[0], dtype=dtype)
        postpro.run(unroll_steps=1)

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
     parser.add_argument("--data_path", type=str)
     parser.add_argument("--loss", type=str)
     parser.add_argument("--devices", type=int, default=1)
     parser.add_argument("--modes", type=int, default=16)
     parser.add_argument("--nodes", type=int, default=1)
     parser.add_argument("--epochs", type=int, default=-1)
     parser.add_argument("--ckpt_path", type=str, default=None)
     parser.add_argument("--precision", type=str, default="32")
     parser.add_argument("--batch_size", type=int, default=4)
     parser.add_argument("--lr", type=float, default=1e-3)
     parser.add_argument("--weight_decay", type=float, default=1e-3)
     parser.add_argument("--seq_len", type=int, nargs='+', default=[10, 100])
     parser.add_argument("--cases", type=str, nargs='+', default=["case1", "case2", "case3"])
     parser.add_argument("--use_affine", action='store_true')
     parser.add_argument("--model", type=str, default='TFNO_t')
     parser.add_argument('--save_path', type=str, default=None)
     parser.add_argument('--overfit_batches', type=int, default=0)
     parser.add_argument('--limit_train_batches', type=float_or_int, default=1.0)
     parser.add_argument("--accumulate_grad_batches", type=int, default=1)
     parser.add_argument("--lr_warmup", action='store_true')
     parser.add_argument("--lr_warmup_steps", type=int, default=5)
     parser.add_argument("--model_config", type=str)
     args = parser.parse_args()
     main(args)