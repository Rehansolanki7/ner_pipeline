# data_pipeline.py
import json

def get_text_from_json(path):
    dataFile = open(path,'r')
    Data = dataFile.read()
    Data = Data.split("\n")
    Data = [item for item in Data if item != '']
    train_data = []
    for item in Data:
        item = json.loads(item)
        train_data.append(item)
    return train_data

def get_labels_with_tokens(path):
    train_data = get_text_from_json(path)
    text = []
    labels = []
    for item in train_data:
        i = 0
        s = item["text"]
        text.append(s)
        tokens = s.split()
        token_tag = []
        prev_s_i = 0
        prev_e_i = len(s)
        for item1 in tokens:
            s_i = s.find(item1, prev_s_i, prev_e_i)
            e_i = s_i + len(item1)
            prev_s_i = s_i
            flag = True
            while(i < len(item["label"])):
                s_i1 = item["label"][i][0]
                e_i1 = item["label"][i][1]
                if(s_i in [s_i1, s_i1-1, s_i1+1] or e_i in [e_i1, e_i1-1, e_i1+1]):
                    token_tag.append([item1, item["label"][i][2]])
                    i = i+1
                    flag = False
                    break
                if(s_i+1 > s_i1 and e_i+1 > e_i1):
                    i = i+1
                break
            if(flag == True):
                token_tag.append([item1, "O"])
        labels.append(token_tag)
    return text, labels


def get_text_and_labels(path):
    text, labels = get_labels_with_tokens(path)
    train_labels = []
    for item in labels:
        temp_label = []
        for item1 in item:
            temp_label.append(item1[1])
        train_labels.append(temp_label)
    return text, train_labels