import torch.nn as nn
import torch
import logging
from transformers import AutoTokenizer,AutoModel

# class Model(nn.Module):
#     def __init__(self,model_name="bert-base-chinese",cache_dir="../.cache",class_num=13,text_ouput_dim=512,dropout=0.25):
#         super().__init__()
#         self.text_ouput_dim = text_ouput_dim
#         self.img_output_dim = 256

#         self.pre_model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir)  # bert
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

#         self.text_head = nn.Sequential(
#             nn.Linear(self.pre_model.config.hidden_size,text_ouput_dim),
#             nn.Dropout(p=dropout),
#         )
        
#         self.img_head = nn.Sequential(  # 降低图片维度
#             nn.Linear(2048, 512),
#             nn.LeakyReLU(512),
#             nn.Dropout(p =dropout),
#             nn.Linear(512,256),
#             nn.LeakyReLU(256),
#             nn.Dropout(p=dropout)
#         )
#         self.cat_dim = self.text_ouput_dim+self.img_output_dim
#         self.output_dim = 512
#         self.classify_head = nn.Sequential(  # 分类头
#             nn.Linear(self.cat_dim, self.output_dim),
#             # nn.BatchNorm1d(self.output_dim),
#             # nn.LeakyReLU(self.output_dim),
#             nn.Dropout(p = dropout),

#             # nn.Linear(512,self.output_dim),
#             # nn.BatchNorm1d(self.output_dim),
#             # nn.LeakyReLU(self.output_dim),
#             # nn.Dropout(p =dropout),
#             nn.Linear(self.output_dim, class_num)
#         )

#     def forward(self,img_features,text_inputs):
#         """
#             text_inputs:dict{"input_ids":Tensor,...}
#             img_features: Tensor
#         """
#         text_emb = self.pre_model(**text_inputs)[1] # (bs,hid_dim) (bs,768)
#         # logging.info(f"text_emb:{text_emb.shape}")
#         text_emb = self.text_head(text_emb)
#         # logging.info(f"text_emb:{text_emb.shape}")
#         img_emb =self.img_head(img_features) #(bs,hid_dim) (bs,512)
#         # logging.info(f"img_emb:{img_emb.shape}")
#         emd = torch.cat([text_emb,img_emb],dim=1)
#         logits = self.classify_head(emd)
#         return logits

#     def tokenize(self,texts):
#         return self.tokenizer(texts,return_tensors="pt", padding="max_length",truncation=True,max_length=32)
#         # 32是测量了数据集中文字最大长度为32


class Model(nn.Module):
    def __init__(self,model_name="bert-base-chinese",cache_dir="../.cache",class_num=13,dropout=0.25):
        super().__init__()

        self.pre_model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir)  # bert
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

        # self.text_output_dim = self.pre_model.config.hidden_size
        self.text_output_dim = 512
        self.text_head = nn.Sequential(
          nn.Linear(self.pre_model.config.hidden_size,self.text_output_dim),
          nn.Dropout(p=dropout)
        )
        self.img_head = nn.Sequential(
          nn.Linear(2048,512),
          nn.Dropout(p=dropout),
          nn.Linear(512,256),
          nn.Dropout(p=dropout),
        )
        self.img_output_dim = 256
        self.cat_dim = self.text_output_dim + self.img_output_dim # 512 + 256
        self.output_dim = 512
        self.classify_head = nn.Sequential(  # 分类头
            nn.Linear(self.cat_dim, self.output_dim),
            nn.Dropout(p = dropout),
            nn.Linear(self.output_dim, class_num)
        )
    def forward(self,img_features,text_inputs):
        """
            text_inputs:dict{"input_ids":Tensor,...}
            img_features: Tensor
        """
        text_emb = self.pre_model(**text_inputs)[1] # (bs,hid_dim) (bs,768)
        # logging.info(f"text_emb:{text_emb.shape}")
        text_emb = self.text_head(text_emb)
        # logging.info(f"text_emb:{text_emb.shape}")
        img_emb = self.img_head(img_features) #(bs,hid_dim) (bs,512)
        # img_emb = img_features
        # logging.info(f"img_emb:{img_emb.shape}")
        emd = torch.cat([text_emb,img_emb],dim=1)
        logits = self.classify_head(emd)
        return logits

    def tokenize(self,texts):
        return self.tokenizer(texts,return_tensors="pt", padding="max_length",truncation=True,max_length=32)
        # 32是测量了数据集中文字最大长度为32