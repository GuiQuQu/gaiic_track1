

# import logging
# 生产负例
import argparse
import json
import os
import random
from tqdm import tqdm
from replace import replace_entry,get_attr_dict
# from train import setup_seed
class_name=['图文','领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度']

class_dict={'图文':['符合','不符合'],
            '领型': ['高领', '连帽', '翻领', '双层领', '西装领', 'U型领', '一字领', '围巾领', '堆堆领', 'V领', '棒球领', '圆领', '斜领', '亨利领'], 
            '袖长': ['短袖', '九分袖', '七分袖', '无袖'], 
            '衣长': ['超短款', '长款', '中长款'], 
            '版型': ['修身型', '宽松型'], 
            '裙长': ['短裙', '中裙', '长裙'], 
            '穿着方式': ['套头', '开衫'], 
            '类别': ['手提包', '单肩包', '斜挎包', '双肩包'], 
            '裤型': ['O型裤', '铅笔裤', '工装裤', '紧身裤', '背带裤', '喇叭裤', '阔腿裤'], 
            '裤长': ['短裤', '五分裤', '七分裤', '九分裤'], 
            '裤门襟': ['松紧', '拉链', '系带'], 
            '闭合方式': ['松紧带', '拉链', '套筒', '系带', '魔术贴', '搭扣'], 
            '鞋帮高度': ['高帮', '低帮']}

"""
    img_dict 映射 img_name to features
    texts {"image_name","text":"xxxx",labels:[0,0,0,1,1...,1]}
"""

def generate_neg_samples(data_path = "data/train_fine.txt",
                        res_data_dir = "data/data_with_neg",
                        attr_dict_path = "data/attr_to_attrvals.json",
                        attr_replace_p = 0.8):
    if not os.path.exists(data_path):
        print("data path not exist")
        return
    if  not os.path.exists(res_data_dir):
        os.makedirs(res_data_dir)
    res_data_path = os.path.join(res_data_dir,"text_data.txt")
    img_dict_path = os.path.join(res_data_dir,"img_dict.txt")
    label_map_path = os.path.join(res_data_dir,"label_map.txt")
    img_dict = {} # 'name': features
    texts = []
    neg_cnt = 0
    posi_cnt = 0
    attr_dict = get_attr_dict(attr_dict_path)
    with open(data_path,"r",encoding="utf-8") as f:
        for line in tqdm(f):
            item_dict = json.loads(line)
            item_dict = replace_entry(item_dict,attr_dict) # 替换等价但是不同的文字
            img_dict[item_dict["img_name"]] = item_dict["feature"]
            match_list = item_dict["match"]
            key_attr_dict = item_dict["key_attr"]
            title = item_dict["title"]
            label = []
            for name in class_name:
                if name in match_list.keys():
                    label+=[1]
                else: 
                    label+=[0]
            text_entry ={
                "img_name":item_dict["img_name"], # 图像feature通过img_name在img_dict查找
                "title":title,
                "label":label,
                "key_attr":key_attr_dict.copy(),
            }
            # print("add res posi:",title)
            texts.append(text_entry)
            posi_cnt+=1
            ### make negative sample ###
            """
                替换策略
                在选中key_attr中的值,将title中的内容替换
                部分替换 ok 
            """
            run_replace = False
            
            label =[]
            for name in class_name[1:]:
                if name in match_list.keys():
                    if random.random() < attr_replace_p:
                        tmp_attr_list = []
                        for attr in class_dict[name]:
                            if attr != key_attr_dict[name]:
                                tmp_attr_list.append(attr)
                        replace_content = random.choice(tmp_attr_list)
                        # print(replace_content)
                        # print(key_attr_dict[name])
                        title = title.replace(key_attr_dict[name],replace_content)
                        key_attr_dict[name] = replace_content
                        run_replace = True
                        label+=[0]
                    else:
                        label+=[1]
                else:
                    label+=[0]
            if run_replace==True:
                label =[0]+label
                text_entry ={
                    "img_name":item_dict["img_name"], # 图像feature通过img_name在img_dict查找
                    "title":title,
                    "label":label,
                    "key_attr":key_attr_dict,
                }
                # print("add res neg:",title)
                texts.append(text_entry)
                neg_cnt+=1
    # print( img_dict_path)
    with open(img_dict_path,"w",encoding="utf-8") as f:
        json_str = json.dumps(img_dict,ensure_ascii=False)
        f.write(json_str)
    with open(res_data_path,"w",encoding="utf-8") as f:
        for item in texts:
            json_str = json.dumps(item,ensure_ascii=False)
            f.write(json_str)
            f.write("\n")
    label_map ={}
    for idx,label in enumerate(class_name):
        label_map[idx] = label
    with open(label_map_path,"w",encoding="utf-8") as f:
        json_str = json.dumps(label_map,ensure_ascii=False)
        f.write(json_str)
    print(f"negative sample size:{neg_cnt},positive_sample size:{posi_cnt},ratio:{neg_cnt/posi_cnt:.3f}")
    
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed",type=int,default=None,help="random seed")
  parser.add_argument("--data-path",type=str,default=None,help="old data path")
  parser.add_argument("--res-datadir",type=str,default=None,help = "res data will be saved in this dir")
  parser.add_argument("--attr-dict-path",type=str,default=None,help="attr_to_attrvals.json path")
  parser.add_argument("--replace-p",type=float,default=0.8,help = "attr replace p")
  args = parser.parse_args()
  if args.seed is not None:
    random.seed(args.seed)
  generate_neg_samples(args.data_path,args.res_datadir,args.attr_dict_path,args.replace_p)
            