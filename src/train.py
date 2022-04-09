import torch
import logging
from model import Model
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(data_loader,criterion,model,optimizer,scheduler,epoch,args,tb_writer=None):
    """
        data_loader: train_dataloader
        model:训练的模型
        optimizer:优化器
        scheduler:调整学习率的优化器
        epoch:这次训练的epoch
        args:参数解析器
        tb_writer:tensorboard记录Writer
    """
    model.train()
    for idx,batch in enumerate(data_loader):

        img_features,texts,labels = batch # (bs,2048),dict,(bs,13)
        img_features = img_features.to(args.device)
        texts = {k:v.squeeze(1).to(device) for k,v in texts.items()}
        
        step = epoch * data_loader.num_batchs + idx
        scheduler(step)
        optimizer.zero_grad()
        
        logtis = model(img_features,texts)
        loss = criterion(logtis,labels)
        loss.backward()
        optimizer.step()

        if args.log_step > 0 and idx % args.log_step == 0:
            "Train Epoch: 1 [12/123](12%)loss:0.111,"
            percent_complete = 100.0 * idx / num_batch_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{idx}/{num_batch_per_epoch}({percent_complete:.1f}%)]\t"
                f"Loss: {loss:.6f}\t"
                f"LR: {optimizer.param_groups[0]['lr']:.5f}"
            )

            timestep = epoch * num_batch_per_epoch + idx
            if tb_writer is not None:
                log_data ={
                    "loss":loss,
                    "lr":optimizer.param_groups[0]['lr'],
                }
                for name,val in log_data.items():
                    name ="train/"+name
                    tb_writer.add_scalar(name,val,timestep)


def evaluate(data_loader,model,criterion,epoch,args,tb_writer=None):
    """
        data_loader:eval_data_loader
        model:模型
        criterion: loss 计算函数
        epoch:该模型对应的epoch
        args:参数解析器
        tb_writer:tensorboard Writer
        评测模型使用的指标
        1. avg loss
        2. 预测准确率
    """
    model.eval()
    total_loss = 0
    num_element = 0
    preds_nums = 0
    pred_true_nums = 0
    with torch.no_grad():
        for batch in data_loader:
            img_features,texts,labels = batch # (bs,2048),dict,(bs,13)
            img_features = img_features.to(args.device)
            texts = {k:v.squeeze(1).to(device) for k,v in texts.items()}
            logits = model(img_features,texts) # (bs,class_num)
            loss = criterion(logtis,labels)
            cumulative_loss += loss
            num_element +=data_loader.batch_size
            total_loss += loss

            preds = (logits.sigmod() > args.threshold).long() # (bs,class_num)
            labels = torch.LongTensor(labels) # (bs,class_num)
            compare_ans = preds == labels
            preds_nums += compare_ans.shape[0] * compare_ans.shape[1]
            pred_true_nums += torch.sum(compare_ans)
    
    metrics ={
        "val_loss":total_loss/num_element,
        "val_acc":pred_true_nums/preds_nums,
        }
    
    logging.info("\t".join([f"{name}:{val:.4f}"  for name,val in metrics.tiems()]))
    
    if tb_writer is not None:
        for name,val in metrics.items():
            name ="evaluate/"+name
            tb_writer.add_scalar(name, scalar_value)
    


