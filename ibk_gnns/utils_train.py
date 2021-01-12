import numpy as np
"""
Some utilities (mostly mirroring keras facilities)
These were prefered to the keras analogues to have greater transparency in what is going on and also have full control of conditions/events etc.
"""
class LossLogger:
    def __init__(self):
        self.losses  = ["loss", "val_loss"]
        self.loss_history = {k:[] for k in self.losses}
    def append_loss(self, loss_):
        self.loss_history['loss'].append(loss_)
        
    def append_val_loss(self, val_loss_):
        self.loss_history['val_loss'].append(val_loss_)
    
    def print(self):
        loss, val_loss = [self.loss_history[vv] for vv in self.losses]
        print("loss: %2.3f, val_loss %2.3f"%(loss[-1], val_loss[-1]))

class EarlyStopping:
    def __init__(self, patience, loss_handle):
        self.patience = patience
        self.loss_handle = loss_handle
    
    def on_epoch_end(self, epoch):
        break_ = False
        
        if len(self.loss_handle) > 1:
            if np.all(np.min(self.loss_handle[-self.patience:]) > self.min_val):
                print("*** Early stopping. ***")
                break_ = True
                
            else:
                self.min_val = np.min(self.loss_handle)
        else:
            
            self.min_val = self.loss_handle[-1]
        
        return break_
    
class LRScheduler:
    def __init__(self, opt_object, base_lr = 0.001,epoch_decay = 50, decay_rate = 0.95, burnin_epochs = 10):
        self.opt_object = opt_object
        self.opt_object.lr.assign(base_lr)
        if burnin_epochs >0:
            self.opt_object.lr.assign(0)

        self.burnin_epochs = burnin_epochs
        self.base_lr = base_lr
        self.epoch_decay = epoch_decay
        self.decay_rate = decay_rate
        
    def on_epoch_end(self,epoch):
        if epoch <=self.burnin_epochs:
            new_lr = self.base_lr * (epoch / float(self.burnin_epochs))
            print("burn-in:  setting lr to %2.5f"%new_lr)
            self.opt_object.lr.assign(new_lr)

        if epoch >= self.epoch_decay:
            lr = self.opt_object.lr.numpy()
            new_lr = lr * self.decay_rate
            print("lr_decay: setting lr to %2.5f"%new_lr)
            self.opt_object.lr.assign(new_lr)
        
