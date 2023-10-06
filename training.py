import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW

def tokenize_adjust_labels(train_text, train_labels, model_path):
    train_text = [item.split() for item in train_text]
    labels = {x for l in train_labels for x in l}
    labels = list(labels)
    train_labels = [[labels.index(lbl) for lbl in sentence_labels] for sentence_labels in train_labels]
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    tokenized_samples = tokenizer(train_text, is_split_into_words=True, padding=True, truncation=True)

    total_adjusted_labels = []
    attention_masks = []

    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = train_labels[k]
        i = -1
        adjusted_label_ids = []
        for wid in word_ids_list:
            if wid is None:
                adjusted_label_ids.append(-100)
            elif wid != prev_wid:
                i = i + 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = wid
            else:
                adjusted_label_ids.append(existing_label_ids[i])

        total_adjusted_labels.append(adjusted_label_ids)
        attention_mask = [1] * len(tokenized_samples["input_ids"][k])
        attention_masks.append(attention_mask)

    tokenized_samples["labels"] = total_adjusted_labels

    return tokenized_samples, attention_masks

def train_model(train_text, train_labels, model_path, n_epoch):
    labels = {x for l in train_labels for x in l}
    labels = list(labels)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(labels))
    train_encoding, train_attention_masks = tokenize_adjust_labels(train_text, train_labels, model_path)
    train_inputs = torch.tensor(train_encoding["input_ids"])
    train_attention_mask = torch.tensor(train_attention_masks)
    train_labels = torch.tensor(train_encoding["labels"])
    train_dataset = TensorDataset(train_inputs, train_attention_mask, train_labels)

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(n_epoch):
        for batch in train_dataloader:
            model.train()
            batch_input_ids = batch[0]
            batch_attention_masks = batch[1]
            batch_labels = batch[2]
            optimizer.zero_grad()
            output = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), r'D:/ner_share/model/ner.model')
    return "model training done and model file saved in the folder"
