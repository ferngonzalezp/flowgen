import torch
import lightning as L
import numpy as np
import torch.nn.functional as F
from neuralop.models import TFNO
from flowgen.utils.loss import nrmse_loss, rH1loss, spec_loss
from flowgen.utils.adaptivePcfLoss import AdaptivePCFLLoss
from flowgen.models.TFNO_t import TFNO_t
from flowgen.models.GL_TFNO_t import GL_TFNO_t
from flowgen.models import Attention_Unet
from flowgen.models.RevIN import RevIN

class tfno(L.LightningModule):
    def __init__(self, loss, lr, modes = 8, precision='full', factorization='tucker', rank=0.42, layers=4, num_classes=3, use_ema=False, affine=False, model='TFNO_t',
     weight_decay=1e-3, lr_warmup = False, lr_warmup_steps = 1000):
        super().__init__()
        self.affine = affine
        self.rev_in = RevIN(6, affine=self.affine)
        self.model_type = model
        self.lr_warmup = lr_warmup
        self.lr_warmup_steps = lr_warmup_steps
        if model=='TFNO':
            self.model = TFNO(n_modes=(modes,modes,modes),
                            hidden_channels=64,
                            n_layers=layers,
                            in_channels= 6,
                            out_channels=6,
                            use_mlp=True,
                            factoization=factorization,
                            rank=rank,
                            norm='instance_norm',
                            fno_block_precision=precision,
                            )
        
        if model=='TFNO_t':
        
            self.model = TFNO_t(modes = (modes,modes,modes), precision=precision, factorization=factorization, 
                    rank=rank, layers=layers, hidden_dim=64, in_channels=6, out_channels=6)
        
        if model=='GL_TFNO':

            self.model = GL_TFNO_t(modes = (modes,modes,modes), precision=precision, factorization=factorization, 
                    rank=rank, layers=layers, hidden_dim=64, in_channels=6, out_channels=6)
        if model=='Attn_UNET':
            self.model = Attention_Unet(6,6, dims=[128,128,128,128], depths=[2,2,2,2])
        
        if use_ema:
            self.ema_model  = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)).requires_grad_(False)
        
        self.loss = loss

        self.epsilon = 1e-1
        self.lr = lr
        self.weight_decay =  weight_decay
        self.val_loss_avg = [0] * num_classes
        self.dynamic_loss = AdaptivePCFLLoss(num_classes=num_classes, gamma=2.0, stability_factor=0.5)
        self.automatic_optimization=True
        self.modes = modes
        self.pf_loss_train = []
        self.pf_steps = 8
        self.ema = use_ema

    def forward(self, inputs, t0, t):

        x = self.rev_in(inputs, 'norm')
        if not self.model=='TFNO':
            x = self.model(x, t0, t)
        else:
            x = self.model(x)
        x = self.rev_in(x, 'denorm')
        return x
    
    def predict(self, inputs, t0, t):
        x = self.rev_in(inputs, 'norm')
        if self.ema:
            if not self.model=='TFNO':
                x = self.ema_model(x, t0, t)
            else:
                x = self.ema_model(x)
        else:
            if not self.model=='TFNO':
                x = self.model(x, t0, t)
            else:
                x = self.model(x)
        x = self.rev_in(x, 'denorm')
        return x
    
    def pushforward_loss(self, y, warmup_steps, time, w=None, reduction=True):
        pred = []
        for time_step in range(y.shape[-1] - warmup_steps - 1):
            if self.model_type == 'GL_TFNO':
                self.model.reset_memeory()
            with torch.no_grad():
                input = y[...,time_step]
                for i in range(warmup_steps):
                    input = self(input, time[...,i + time_step], time[...,i+1+time_step])
                    if self.model_type == 'GL_TFNO':
                        self.model.detach_memeory()
            pred.append(self(input,time[...,time_step+warmup_steps], time[...,time_step+warmup_steps+1]))
        pred = torch.stack(pred,dim=-1)
        nrmse = spec_loss(pred, y[...,warmup_steps+1:], w=w, reduction=reduction)
        h1_loss= rH1loss(pred,y[...,warmup_steps+1:], w=w, reduction=reduction)
        loss = torch.clamp(h1_loss + nrmse, min=1e-8, max=10)

        #print(f"y_true range: {pred.min().item()} to {pred.max().item()}")
        #assert not torch.isnan(pred).any(), "NaN in pred of pushforward"
        return loss
    
    def causality_loss(self, y, time):
        
        input = y[...,0]
        pred = []
        for i in range(y.shape[-1]-1):
            pred.append(self(input, time[...,i], time[...,i+1]))
            input = pred[-1]
        pred = torch.stack(pred, dim=-1)
        
        # update w
        w = []
        L = []
        
        with torch.no_grad():
            
            for i in range(pred.shape[-1]):
                L.append(nrmse_loss(pred[...,i], y[...,1+i]) + 
                        rH1loss(pred[...,i], y[...,1+i]))
                w.append(torch.exp(-self.epsilon * sum(L[:i+1]) ))
            if min(w) > 0.99 and self.epsilon < 100:
                self.epsilon *= 10
                
        #computed weighted loss
        loss = 0.0
        for i in range(pred.shape[-1]):
                loss += w[i] * nrmse_loss(pred[...,i], y[...,1+i]) + w[i] * rH1loss(pred[...,i], y[...,1+i])
        for i in range(len(w)):
            self.log_dict({'w_{}'.format(i+1): w[i].detach()}, prog_bar=True)
        return loss / (i+1)

    def training_step(self, batch, batch_idx):

        #with torch.autograd.detect_anomaly():
            if self.automatic_optimization==False:
                sch = self.lr_schedulers()
                opt = self.optimizers()

            y = batch[0]
            time =  batch[-1]
            y_pred = []
            if self.model_type == 'GL_TFNO':
                self.model.reset_memory()
            for i in range(y.shape[-1]-1):
                y_pred.append(self(y[...,i], time[...,i], time[...,i+1]))
                if (i + 1) % 5 == 0:
                    if self.model_type == 'GL_TFNO':
                        self.model.detach_memory()
            y_pred = torch.stack(y_pred, dim=-1)
            if self.loss == 'one_step':
                nrmse = nrmse_loss(y_pred, y[...,1:])
                h1_loss= rH1loss(y_pred,y[...,1:])
                loss = h1_loss + nrmse
                values = {'nrmse': nrmse, 'H1_loss': h1_loss}
            elif self.loss == 'pushforward':
                nrmse = spec_loss(y_pred, y[...,1:])
                h1_loss= rH1loss(y_pred,y[...,1:])
                warmup_steps = (batch_idx + 1) % self.pf_steps + 1
                pf_loss = self.pushforward_loss(y, warmup_steps, time)
                self.pf_loss_train.append(pf_loss)
                loss = h1_loss + nrmse + pf_loss
                values = {'nrmse': nrmse, 'H1_loss': h1_loss, 'PF_loss': pf_loss}
            elif self.loss == 'causality':
                loss = self.causality_loss(y, time)
                values = {'causality_loss': loss}
            elif self.loss == 'dynamic':
                labels = batch[2]
                loss_func = lambda y_pred, y, w, reduction: nrmse_loss(y_pred, y[...,1:], w=w, reduction=reduction) + rH1loss(y_pred,y[...,1:], w=w, reduction=reduction) + self.pushforward_loss(y, 3, w=w, reduction=reduction)
                loss = self.dynamic_loss(y_pred, y, labels, loss_func)
                values = {'dynamic_loss': loss}
                for i, w in enumerate(self.dynamic_loss.weights):
                        self.log('weights_{}'.format(i+1), w, prog_bar=True)
            self.log_dict(values, prog_bar=True)

            if self.automatic_optimization==False:
                self.manual_backward(loss/4)

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in {name}")


                if (batch_idx + 1) % 4 == 0:
                    # clip gradients
                    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

                    opt.step()
                    opt.zero_grad()
                
                # step every N epochs
                if self.trainer.is_last_batch:
                    sch.step()
            return loss
    
    def on_train_epoch_end(self):
        if self.loss == 'pushfroward':
            pf_loss_mean = torch.stack(self.pf_loss_train).mean()
            #if pf_loss_mean < 2e-1 and self.pf_steps < 8:
            if (self.trainer.current_epoch + 1) % 25 == 0 and self.pf_steps < 8:
                self.pf_steps += 1
            print(self.pf_steps)
            self.pf_loss_train.clear()
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    # Add your custom logic to run directly before `optimizer.step()`

        optimizer.step(closure=optimizer_closure)

    # Add your custom logic to run directly after `optimizer.step()`
        if self.ema:
            self.ema_model.update_parameters(self.model)
                          
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch[0]
        y_pred = []
        input = y[...,0]
        time = batch[-1]
        if self.model_type=='GL_TFNO':
            self.model.reset_memory()
        for i in range(y.shape[-1]-1):
            pred = self.predict(input, time[...,i], time[...,i+1])
            y_pred.append(pred)
            input = pred
        y_pred = torch.stack(y_pred, dim=-1)
        nrmse = nrmse_loss(y_pred, y[...,1:])
        self.val_loss_avg[dataloader_idx] += nrmse
        #self.log('val_loss_avg', nrmse, prog_bar=False, sync_dist=True)
        self.log_dict({'val_loss_case{}'.format(dataloader_idx): nrmse,}, prog_bar=True, sync_dist=True)
            
                          
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_warmup:
            scheduler1 =  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=self.lr_warmup_steps)
            steps = max(self.trainer.max_epochs, self.trainer.max_steps)
            if self.trainer.max_epochs > self.trainer.max_steps:
                steps = self.trainer.estimated_stepping_batches
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * self.trainer.num_training_batches, eta_min=1e-6)
            scheduler3 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1e-6 / self.lr, total_iters=steps)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], milestones=[sef.lr_warmup_steps, 100 * self.trainer.num_training_batches])
        else:
            steps = max(self.trainer.max_epochs, self.trainer.max_steps)
            if self.trainer.max_epochs > self.trainer.max_steps:
                steps = self.trainer.estimated_stepping_batches
            scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-6)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            #"scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, eta_min=1e-6, T_mult=2),
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        #return optimizer
        return  {"optimizer": optimizer, 
                 "lr_scheduler": lr_scheduler_config,
                 }


