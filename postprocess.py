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
from UNET_shub import Unet
import os

            
def main(args):
    data_path = args.data_path #Path to dataset location
    train_data_dirs = [data_path+f"train/{case_name}" for case_name in args.cases]
    val_dir = [data_path+f"val/{case_name}" for case_name in args.cases]
    print(val_dir)
    dm = hitOfflineDataModule(val_dirs = val_dir, data_dir = train_data_dirs, batch_size =  args.batch_size, seq_len=args.seq_len)
    affine=False
    if args.use_affine:
        affine=True

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
        

    print(save_path)
    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    if args.model == 'UNET_shub':
        model = Unet.load_from_checkpoint(args.ckpt_path, loss=args.loss, lr=args.lr)
        model = model.predict
    else:
        model = tfno.load_from_checkpoint(args.ckpt_path, loss=args.loss, lr=args.lr, precision='full', modes=16, num_classes=len(train_data_dirs), model=args.model)
    dm.setup('fit')
    postpro = Postprocess(dm.val_dataloader(), model, save_path, device, val_dir[0], dtype=args.dtype)
    postpro.run(unroll_steps=args.unroll_steps)


if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument('--data_path', type=str)
     parser.add_argument("--loss", type=str)
     parser.add_argument("--devices", type=int, default=1)
     parser.add_argument("--modes", type=int, default=16)
     parser.add_argument("--nodes", type=int, default=1)
     parser.add_argument("--epochs", type=int, default=-1)
     parser.add_argument("--ckpt_path", type=str, default=None)
     parser.add_argument("--batch_size", type=int, default=4)
     parser.add_argument("--unroll_steps", type=int, default=1)
     parser.add_argument("--lr", type=float, default=1e-3)
     parser.add_argument("--seq_len", type=int, nargs='+', default=[10, 100])
     parser.add_argument("--use_affine", action='store_true')
     parser.add_argument("--model", type=str, default='TFNO_t')
     parser.add_argument('--save_path', type=str, default=None)
     parser.add_argument("--cases", type=str, nargs='+', default=["case1", "case2", "case3"])
     parser.add_argument("--dtype", type=str, default=None)
     args = parser.parse_args()
     main(args)