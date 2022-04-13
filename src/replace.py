# 替换等价文字
import json
import os

def get_attr_dict(attr_path):

    # attr_path ="data/attr_to_attrvals.json"

    if  not os.path.exists(attr_path):
        print("attr_path not exists")
        return None
    attr_dict ={}
    with open(attr_path,"r",encoding="utf-8") as f:
       tmp_dict = json.loads(f.read())
    for attr_name,attr_list in tmp_dict.items():
        attr_list = [ attr_content.split("=") for attr_content in attr_list]
        attr_dict[attr_name] = attr_list
    # print(attr_dict)
    return attr_dict

# attr_dict = get_attr_dict()

def replace_entry(data_entry,attr_dict):
    """
        替换title和key_attr
    """
    if attr_dict is None:
        raise ValueError("no attr_dict")
    if "key_attr" not in data_entry.keys() or "title" not in data_entry.keys():
        raise ValueError("wrong format data entry")
    old_key_attr = data_entry["key_attr"]  # "key_attr": {"裤门襟": "松紧"}
    new_key_attr ={}
    title = data_entry["title"] 
    for key,val in old_key_attr.items():
        attr_val_list = attr_dict[key]
        for attr_vals in attr_val_list:
            if val in attr_vals:
                new_key_attr[key] = attr_vals[0]
    # print(new_key_attr)
    data_entry["key_attr"] = new_key_attr
    for key in old_key_attr:
        old_val = old_key_attr[key]
        new_val = new_key_attr[key]
        if old_val!=new_val:
            title = title.replace(old_val,new_val)
    data_entry["title"] = title
    return data_entry