import torch
from torch.cuda.amp import autocast as autocast
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


def train(data_loader,criterion,model,optimizer,scheduler,scaler,epoch,args,tb_writer=None):
    """
        data_loader: train_dataloader
        model:训练的模型
        optimizer:优化器
        scheduler:调整学习率的优化器
        scaler:混合精度训练模型
        epoch:这次训练的epoch
        args:参数解析器
        tb_writer:tensorboard记录Writer
    """
    model.train()
    device = args.device
    num_batch_per_epoch = data_loader.num_batches
    for idx,batch in enumerate(data_loader):

        img_features,texts,labels = batch # (bs,2048),dict,(bs,13)
        img_features = img_features.to(args.device)
        labels = torch.stack(labels,dim=1).to(device)
        texts = {k:v.squeeze(1).to(device) for k,v in texts.items()}
        
        step = epoch * data_loader.num_batches + idx
        scheduler(step)
        optimizer.zero_grad()
        # 混合精度训练
        if args.device.startswith("cuda") and scaler is not None:
            with autocast():
                logits = model(img_features,texts)
                loss = criterion(logits,labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # float32精度训练
            logits = model(img_features,texts)
            loss = criterion(logits,labels)
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
    total_loss = 0  # 所有batch的loss和
    # num_element = 0
    preds_nums = 0 # 总共进行的预测的个数
    pred_true_nums = 0 # 所有真实标签结果为1的分类标签的个数
    pred_posi_true_nums = 0 # 真实标签为1,预测结果也为1的预测个数
    labels_posi_nums = 0 # 
    with torch.no_grad():
        for batch in data_loader:
            img_features,texts,labels = batch # (bs,2048),dict,(bs,13)
            img_features = img_features.to(args.device)
            labels = torch.stack(labels,dim=1).to(args.device)
            texts = {k:v.squeeze(1).to(args.device) for k,v in texts.items()}
            logits = model(img_features,texts) # (bs,class_num)

            loss = criterion(logits,labels)
            total_loss += loss
            # num_element += data_loader.batch_size
            total_loss += loss

            preds = (torch.sigmoid(logits) > args.threshold).long() # (bs,class_num)
            # labels = torch.LongTensor(labels) # (bs,class_num)
            compare_ans = preds == labels
            preds_nums += compare_ans.shape[0] * compare_ans.shape[1]
            pred_true_nums += torch.sum(compare_ans)
            labels_posi_nums += torch.sum(labels)
            pred_posi_true_nums += torch.sum(preds+labels==2)
    
    metrics ={
        "val_loss":total_loss/data_loader.num_batches,
        "val_acc":pred_true_nums/preds_nums,
        "posi_val_acc": pred_posi_true_nums/labels_posi_nums,
        }
    
    logging.info(f"Epoch:{epoch}: "+"\t".join([f"{name}:{val:.4f}"  for name,val in metrics.items()]))
    
    if tb_writer is not None:
        for name,val in metrics.items():
            name ="evaluate/"+name
            tb_writer.add_scalar(name, val)
    


