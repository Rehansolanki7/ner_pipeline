# prediction.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import copy
from faker import Faker

def get_model(model_path, fine_tune_model_path, labels):
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels = len(labels))
    # model.to(device)
    model.load_state_dict(torch.load(fine_tune_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
        
def tag_sentence(model_path, fine_tune_model_path, labels, text):
    id2tag = {id: tag for id, tag in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(text.split(), is_split_into_words=True,padding=True,truncation=True, return_tensors="pt")
    model = get_model(model_path, fine_tune_model_path, labels)
    outputs = model(**inputs)
    probs = outputs[0][0].softmax(1)
    word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()]) 
                    for i, tagid in enumerate (probs.argmax(axis=1))]
    return pd.DataFrame(word_tags, columns=['word', 'tag'])

def final_output(model_path, fine_tune_model_path, labels, text):
    res = tag_sentence(model_path, fine_tune_model_path, labels, text)
    res1 = res.iloc[1:len(res)-1]
    res1 = list(zip(res1["word"], res1["tag"]))
    words = copy.deepcopy(res1)
    new_words = []
    i = 0
    while i < len(words):
        if words[i][1] != "O":
            word = words[i][0]
            j = i + 1
            while j < len(words) and words[j][1] == words[i][1] and words[j][0].startswith("##"):
                word += words[j][0][2:]
                j += 1
            new_words.append((word, words[i][1]))
            i = j
        else:
            new_words.append(words[i])
            i += 1
    output = []
    prev_label = None
    for token, label in new_words:
        if label == prev_label and label != "O":
            output[-1] = (output[-1][0] + token, label)
        else:
            output.append((token, label))
        prev_label = label
    return output

def get_masked_data(text, ner_result):
    
    fake = Faker()
    text1 = copy.deepcopy(text)
    text1 = text1.lower()    
    for item in ner_result:
        if("Name" in item[1]):
            mask_name = fake.name()
            mask_name = mask_name.split()
            text1 = text1.replace(item[0], mask_name[0])
        elif("Id" in item[1]):
            mask_id = fake.random_number(digits=8)
            text1 = text1.replace(item[0], str(mask_id))
        elif("Phone" in item[1]):
            mask_phone = fake.phone_number()
            text1 = text1.replace(item[0], str(mask_phone))
    return text1
