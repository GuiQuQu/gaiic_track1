

from time import gmtime,strftime
import os

from train import setup_seed,train,evaluate
from model import Model
from dataset import create_dataloader
from params import parse_args
from logger import set_logger
from scheduler import consine_lr
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim


def train_worker(args):
    if args.log_path is None:
        print("Error,no log path,Use -- log-path {} to set log_path")
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
   
    tensorboard_path = os.path.join(args.log_path,args.name,"tensorboard")
    checkpoint_path = os.path.join(args.log_path,args.name,"checkpoint")
    if args.tb:
        tmp_path_list = [ tensorboard_path,checkpoint_path]
    else:
        tmp_path_list = [checkpoint_path]
    for dirname in tmp_path_list:
        if  not os.path.exists(dirname):
            os.makedirs(dirname,exist_ok=True) # exist_ok设为true ,目标文件夹存在也不会报OSError
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
        
    model =Model()
    model.to(ars.device)
    tokenize = lambda x : model.tokenize(x)
    dls = create_dataloader(args, tokenize)

    name_paramters = list(model.named_parameters())
    paramters = [p for n,p in name_paramters]
    optimizer = optim.AdamW(
            [
            {"params":paramters,"weight decay":0.}
            ],
        lr=args.lr,
        betas=(args.beta1,args.beta2),
        eps=args.eps)
    steps = args.epochs * dls["train"].num_batches
    scheduler = consine_lr(optimizer, args.lr, args.warmup, steps)
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(args.device)
            optimizer.load_state_dict(checkpoint["optimzer"])
            start_epoch = checkpoint["epoch"]
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    tb_writer =None
    if args.tb:
        tb_writer = SummaryWriter(tensorboard_path)  

    criterion = torch.nn.MultiLabelSoftMarginLoss()

    for epoch in range(start_epoch,args.epochs):
        train(dls["train"],criterion,model,optimizer,scheduler,epoch,args,tb_writer)
        evaluate(dls["eval"],model,criterion,epoch,args,tb_writer)
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
    if args.test_file is None:
        print("Error,Check -- test-file {}")
        return -1
    


def main():
    args = parse_args()
    if args.is_train:
        train_worker(args)
    else:
        predict_workers(args)