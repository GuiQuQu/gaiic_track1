
import numpy as np

def _assign_lr(optimizer,new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr,warmup_length,step):
    return base_lr *  (step+1) / warmup_length

def consine_lr(optimizer,base_lr,warmup_length,steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr,warmup_length,step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * base_lr * (1 + np.cos(e/es * np.pi)) 
        _assign_lr(optimizer, lr)
    return _lr_adjuster     
