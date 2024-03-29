
from torch.utils.data import Dataset,DataLoader,random_split
import torch
import numpy as np

from tqdm import tqdm
import json

import os

class TrainDataSet(Dataset):
    def __init__(self,input_filename,img_dict_path,tokenize,is_train):
        """
            input_filename 训练文件
            tokenize: token化的函数
        """
        self.tokenize = tokenize
        self.img_dict ={}
        self.items = [] # {"img_name":"",'title':'xxxx','labels':[1,0,0,0],"key_attr": {"裤型": "喇叭裤", "裤长": "九分裤"}}
        for file in input_filename.split(","):
            with open(file,"r",encoding="utf-8") as f:
                for line in tqdm(f):
                    json_dict = json.loads(line)
                    self.items.append(json_dict)
        with open(img_dict_path,"r",encoding='utf-8') as f:
            self.img_dict = json.loads(f.read())
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name = self.items[idx]["img_name"]
        img_feature = np.array(self.img_dict[img_name]).astype(np.float32)  # (2048)
        title  = self.items[idx]['title']
        label = self.items[idx]['label']
        return torch.from_numpy(img_feature),self.tokenize(title),label


class TestDataSet(Dataset):
    def __init__(self,test_file_path,class_map_path,tokenize):
        self.items = []
        self.tokenize = tokenize 
        with open(class_map_path,"r",encoding="utf-8") as f:
            tmp = json.loads(f.read()) # idx->class_num
            self.idx_2_class = {int(k):v for k,v in tmp.items()}
        # self.class_2_idx = {v:k for k,v in self.idx_2_class}
        with open(test_file_path,"r",encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
                
    def __len__(self):
        return len(self.items)

    def __getitem__(self,idx): # (bs,2048) dict [[...],[...]]
        title = self.items[idx]["title"]
        query = self.items[idx]["query"]
        img_feature = np.array(self.items[idx]["feature"]).astype(np.float32)
        img_name = self.items[idx]["img_name"]
        need_query = [] # 存在该query加1,不存在加0
        query = set(query)
        for idx,class_name in self.idx_2_class.items():
            if class_name in query:
                need_query.append(1)
            else:
                need_query.append(0)
        return img_name,torch.from_numpy(img_feature),self.tokenize(title),need_query



def create_dataloader(args,tokenize):
    """
        args.train_dir
        args.is_train
        args.train_ratio
        args.batch_size
        args.workers
    """
    if args.is_train:
        input_filename = os.path.join(args.train_dir,"text_data.txt")
        img_dict_path = os.path.join(args.train_dir,"img_dict.txt")
        dataset = TrainDataSet(input_filename, img_dict_path, tokenize, args.is_train)
        train_len = int(len(dataset) * args.train_ratio)
        eval_len = len(dataset) - train_len
        train_ds,eval_ds = random_split(dataset, [train_len,eval_len])
        dls = {
            "train":DataLoader(train_ds,batch_size = args.batch_size,shuffle=True, num_workers=args.workers,pin_memory=False,drop_last=True),
            "eval":DataLoader(eval_ds,batch_size = args.batch_size,shuffle=False, num_workers=args.workers,pin_memory=False,drop_last=True)
        }
        for k,dl in dls.items():
            dl.num_samples = len(dataset)
            dl.num_batches = len(dl)
        return dls
    else:
        # 加载测试数据集或者评测数据集
        test_file_path = args.test_file
        class_map_path =args.class_map
        test_dataset =TestDataSet(test_file_path, class_map_path, tokenize)
        dls = {
            "test":DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,drop_last=False)
        }
        for k,dl in dls.items():
            dl.num_batches =len (dl)
        return dls,test_dataset.idx_2_class


