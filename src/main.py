

from time import gmtime,strftime
import os
import json

import logging
from tqdm import tqdm

from train import setup_seed,train,evaluate
from model import Model
from dataset import create_dataloader
from params import parse_args
from logger import set_logger
from scheduler import consine_lr

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim

import numpy as np


def train_worker(args):
    if args.is_train is False:
      print("please set --is-train {} is True")
      return -1
    if args.logs is None:
        print("Error,no log path,Use -- logs {} to set log_path")
        return -1
    if args.train_dir is None:
        print("Error,no train data,use --train-dir {} to set train data dir")
        return -1
    if args.batch_size is None:
        print("Error,Use --batch-size {} to set batch size")
        return -1
    if args.lr is None or args.beta1 is None or args.beta2 is None or args.eps is None:
        print("Error,Check --lr {},--beta1 {},--beta2 {},--eps {}")
        return -1
    if args.warmup is None:
        print("Error,Check --warmup {}") 
        return -1
    if args.epochs is None:
        print("Error,Check --epochs {}")
        return -1
    if args.name is None:
        args.name = strftime(f"lr={args.lr}_batch-size={args.batch_size}_date=%Y-%m-%d-%H-%M-%S",gmtime())
   
    args.tensorboard_path = os.path.join(args.logs,args.name,"tensorboard")
    args.checkpoint_path = os.path.join(args.logs,args.name,"checkpoint")
    if args.tb:
        tmp_path_list = [ args.tensorboard_path,args.checkpoint_path]
    else:
        tmp_path_list = [args.checkpoint_path]
    for dirname in tmp_path_list:
        if  not os.path.exists(dirname):
            os.makedirs(dirname,exist_ok=True) # exist_ok设为true ,目标文件夹存在也不会报OSError
    # log
    args.log_path = os.path.join(args.logs,args.name,"out.log") 
    log_level = logging.DEBUG if args.debug else logging.INFO
    set_logger(args.log_path, log_level)

    # device
    logging.info(f"Use {args.device} train")

    # args prarms save
    param_txt = os.path.join(args.logs,args.name,"params.txt")
    with open(param_txt,"w",encoding="utf-8") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            f.write(f"{name}: {val}\n")

    if args.seed is not None:
        setup_seed(args.seed)
    #
    if args.cache_dir is not None:
      model = Model(cache_dir=args.cache_dir)
    else:
      model = Model()
    model.to(args.device)
    tokenize = lambda x : model.tokenize(x)
    dls = create_dataloader(args, tokenize)

    name_paramters = list(model.named_parameters())
    paramters = [p for n,p in name_paramters]
    optimizer = optim.AdamW(
            [
            {"params":paramters,"weight decay":args.wd}
            ],
        lr=args.lr,
        betas=(args.beta1,args.beta2),
        eps=args.eps)
    steps = args.epochs * dls["train"].num_batches
    if aegs.warmup is not None:
        scheduler = consine_lr(optimizer, args.lr, args.warmup, steps)
    else:
        scheduler =optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2,eta_min=5e-6,last_epoch=-1)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(args.device)
            optimizer.load_state_dict(checkpoint["optimzer"])
            start_epoch = checkpoint["epoch"]
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    tb_writer = None
    if args.tb:
        tb_writer = SummaryWriter(args.tensorboard_path)  

    criterion = torch.nn.MultiLabelSoftMarginLoss()

    for epoch in range(start_epoch,args.epochs):
        # train(dls["train"],criterion,model,optimizer,scheduler,epoch,args,tb_writer)
        # evaluate(dls["eval"],model,criterion,epoch,args,tb_writer)
        # save checkpoint
        if args.save_frequency > 0 and (epoch + 1) % args.save_frequency == 0:
            torch.save({
                "epoch":epoch + 1,
                "name":args.name,
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.checkpoint_path,f"epoch_{epoch+1}.pt"))
    
    if args.save_final:
            torch.save({
                "epoch":args.epochs,
                "name":args.name,
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.checkpoint_path,f"epoch_{args.epochs}.pt"))
        
def predict_workers(args):

    if args.test_file is None or args.class_map is None:
        print("Error,Check --test-file {} --class-map {}")
        return -1
    if args.is_train is True:
        print("please set --is-train {} is False")
        return -1
    if args.resume is None:
        print("Error,Check --resume {} ")
        return -1
    if args.pred_res_path is None:
        print("Error,Check --pred-res-path {} ")
    if args.seed is not None:
        setup_seed(args.seed)

    log_dir = os.path.join(".","preds")
    if  not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    log_file = os.path.join("preds","pred.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    set_logger(log_file, log_level)    

    logging.info(f"Use {args.device} to predict")
    if args.cache_dir is not None:
      model = Model(cache_dir=args.cache_dir)
    else:
      model = Model()
    model.to(args.device)
    tokenize = lambda x : model.tokenize(x)
    
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(args.device)
            start_epoch = checkpoint["epoch"]
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    with open(args.class_map,"r",encoding="utf-8") as f:
        idx2class = json.loads(f.read())    
    class2idx = {v:int(k) for k,v in idx2class.items()}
    preds_res = []
    model.eval()
    with torch.no_grad():
        with open(args.test_file,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                item = json.loads(line)
                img_name = item["img_name"]
                query = item["query"]
                img_features = torch.from_numpy(np.array(item["feature"]).astype(np.float32)).view(1,-1).to(args.device) # (1,2048)
                texts = tokenize(item["title"])
                texts = {k:v.to(args.device) for k,v in texts.items()}
                logits = model(img_features,texts) # (bs,class_num)
                preds = (torch.sigmoid(logits) > args.threshold).long().squeeze(0) # (1,13)
                preds = preds.detach().cpu().tolist()
                match = {}
                for q in query:
                    match[q] = preds[class2idx[q]]
                preds_res.append({"img_name":img_name,"match":match})
            
    with open(args.pred_res_path,"w",encoding="utf-8") as f:
        for result in preds_res:
            json_str = json.dumps(result,ensure_ascii=False)
            f.write(json_str)
            f.write("\n")

def main():
    args = parse_args()
    if args.is_train:
        train_worker(args)
    else:
        predict_workers(args)

if __name__=='__main__':
  main()