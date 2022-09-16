import sys

import torch
import numpy as np
import os
import random
import copy
import pickle
import ipdb
from tqdm import tqdm
import json

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def load_jsonl(file):
    data = {}
    idx = 0
    with open(file) as json_file:
        for line in json_file:
            data[idx] = json.loads(line)
            idx += 1
    return data

def load_jsonl_list(file):
    data = []
    with open(file) as json_file:
        for line in json_file:
            data.append(json.loads(line)) 
    return data

def load_raw_jsonl(file):
    json_file = open(file)
    return json_file

def flattening_tensor(batch_key):
    try:
        result = torch.stack(batch_key)
        return result.transpose(1,0)
    except:
        return batch_key
    
def processing_dataloader(batch):
    processed_batch = {}
    for key in batch.keys():
        processed_batch[key] = flattening_tensor(batch[key])
    return processed_batch

def model_save(model, optimizer, save_path):
    file = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}
    torch.save(file, save_path)
    return

def dict_to_device(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict

def correct_or_not(output, gt):
    predict = output.logits.softmax(dim=1).argmax(1)
    result = (predict == gt)
    if False in result.cpu().detach().numpy():
        correct = 1
    else:
        correct = 0
    return correct

def truncation(context, hypothesis, max_length):
    """Truncates a context in place to the maximum length."""
    # This is a function to truncate context which is usually much longer than hypothesis
    
    while True:
        total_length = len(context) + len(hypothesis)
        if total_length <= max_length:
            break
        if len(context) >= len(hypothesis):
            del context[-2]
        else:
            del hypothesis[-2]
    return context, hypothesis

def convert_label(label):
    label_map = {'e':0, 'n':1, 'c':2}
    label_id = label_map[label]
    return label_id

def label_onehot(label):
    onehot_map = {0:torch.tensor([1,0,0]), 1: torch.tensor([0,1,0]), 2:torch.tensor([0,0,1])}
    return onehot_map[label]

def csv_write(csv_file, flag, fieldnames = None, data = None):
    if fieldnames != None:
        #open new csv file for saving results
        if flag == 'w':
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                f.close()
        
        #original
        elif flag == 'a':
            #create dict
            write_dict = {}
            for idx, fieldname in enumerate(fieldnames):
                write_dict[fieldname] = data[idx]

            with open(csv_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(write_dict)
                f.close()
    else:
        if flag == 'w':
            with open(csv_file, 'w', newline='') as f:
                #writer = csv.writer(f)
                f.close()
        
        #original
        elif flag == 'a':
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
                f.close()		
    return 