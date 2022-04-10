import argparse

default_optim_args ={
    "lr":1e-4,
    "beta1":0.9,
    "beta2":0.999,
    "eps":1e-8,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=None,help="random seed")
    parser.add_argument("--name",type=str,default=None,help="experiment name")
    parser.add_argument("--is-train",type=bool,default=True,help="it`s true if train")
    # log
    parser.add_argument("--logs",type=str,default="./log",help="logger file path")
    parser.add_argument("--debug",type=bool,default=False,help="debug is true means that more info will be reported")
    # data path
    parser.add_argument("--train-dir",type=str,default=None,help="training data dir")
    parser.add_argument("--train-ratio",type=float,default=0.9,help="split ratio of train data and eval data")
    parser.add_argument("--batch-size",type=int,default=None,help="batch-size")
    parser.add_argument("--workers",type=int,default=0,help="'num_workers' param of DataLoader")
    # train
    parser.add_argument("--epochs",type=int,default=0,help="epoch")
    parser.add_argument("--log-step",type=int,default=0,help="log train result after steps")
    parser.add_argument("--save-frequency",type=int,default=0,help="save-frequency")
    parser.add_argument("--save-final",type=bool,default=True,help="save final model")
    parser.add_argument("--tb",type=bool,default=True,help="whether or not use tensorboard")
    parser.add_argument("--device",type=str,default="cpu",help="train device")

    # optimizer
    parser.add_argument("--lr",type=float,default=None,help="learning rate")
    parser.add_argument("--beta1",type=float,default=None,help="AdamW beta1")
    parser.add_argument("--beta2",type=float,default=None,help="AdamW beta2")
    parser.add_argument("--eps",type=float,default=None,help="AdamW eps")
    parser.add_argument("--warmup",type=float,default=None,help="warmup length")
    parser.add_argument("--wd",type=float,default=0.2,help="weight deacy")
    # predict
    parser.add_argument("--threshold",type=float,default=0.5,help="predict threshold")
    parser.add_argument("--test-file",type=str,default=None,help="test file path")

    # pretrained_model
    parser.add_argument("--cache-dir",type=str,default=None,help ="pretrained model download path")
    # load model and going on train or predict
    parser.add_argument("--resume",type=str,default=None,help="existed model path")
    args = parser.parse_args()

    for name,val in default_optim_args.items():
      if getattr(args, name) is None:
        setattr(args, name, val)

    
    return args