import torch.nn as nn
import torch

from transformers import AutoTokenizer,AutoModel

class Model(nn.Module):
    def __init__(self,model_name="bert-base-chinese",class_num=13,text_ouput_dim=512,dropout=0.25):
        super().__init__()
        self.text_ouput_dim = text_ouput_dim
        self.img_output_dim = 256

        self.pre_model = AutoModel.from_pretrained(model_name,cache_dir="../.cache")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='../.cache')
        self.text_head = nn.Sequential(
            nn.Linear(self.pre_model.config.hidden_size,text_ouput_dim),
            nn.Dropout(p=dropout),
        )
        self.img_model = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(512),
            nn.Dropout(p =dropout),
            nn.Linear(512,256),
            nn.LeakyReLU(256),
            nn.Dropout(p=dropout)
        )
        self.cat_dim = self.text_ouput_dim+self.img_output_dim
        self.output_dim = 512
        self.classify_head = nn.Sequential(
            nn.Linear(self.cat_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(self.output_dim),
            nn.Dropout(p = dropout),

            # nn.Linear(512,self.output_dim),
            # nn.BatchNorm1d(self.output_dim),
            # nn.LeakyReLU(self.output_dim),
            # nn.Dropout(p =dropout),
            nn.Linear(self.output_dim, class_num)
            
        )

    def forward(self,text_inputs,img_features):
        """
            text_inputs:dict{"input_ids":Tensor,...}
            img_features: Tensor
        """
        text_emb = self.text_model(**text_inputs)[0] # (bs,hid_dim) (bs,768)
        img_emb =self.img_model(img_features) #(bs,hid_dim) (bs,512)
        emd = torch.cat([text_emb,img_emb],dim=1)
        logits = self.classify_head(emd)
        return logits

    def tokenize(self,texts):
        return self.tokenizer(texts,return_tensors="pt", padding="max_length",truncation=True,max_length=32)
        # 32是测量了数据集中文字最大长度为32