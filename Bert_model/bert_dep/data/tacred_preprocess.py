"""
Data loader for TACRED json files.
"""
import sys
sys.path.append("/work/relation_extraction/Bert_model/baseline/data")

import json
import random
import torch
from tqdm import tqdm
import numpy as np
from pytorch_transformers.tokenization_bert import BertTokenizer
from allennlp.data.tokenizers import Token
from bert_tokenize import bert_tokenize

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, entity_type, relation_type, opt=False):

        with open(filename) as infile:
            raw_data = json.load(infile)
            
        self.raw_data = raw_data
        data = self.preprocess(raw_data, entity_type, relation_type, opt)
        self.data = data

    def preprocess(self, data, entity_type, relation_type, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        e_type2idx = {t:i for i,t in enumerate(entity_type)}
        r_type2idx = {t:i for i,t in enumerate(relation_type)}
        
        # load pretrain bert
        if opt["lower"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            
        # iter through all data
        tqdm_data = tqdm(data)
        for d in tqdm_data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            
            # bert tokenizer
            bert_tokenize(tokenizer, d, opt)
            anc_idx, seq_idx = find_lowest_ancest(d["subj_start"], d["obj_start"], d["stanford_head"])
            
            seq_len = len(seq_idx) + 2
            tokens = []
            pos = []
            ner = []
            deprel = []
            
            positions = get_positions(1, 1, seq_len - 2, seq_len - 2, e_type2idx[d["subj_type"]], e_type2idx[d["obj_type"]], seq_len)
            
            for i,idx in enumerate(seq_idx):
                tokens.append(d["token"][idx])
                pos.append(d["stanford_pos"][idx])
                ner.append(d["stanford_ner"][idx])
                deprel.append(d["stanford_deprel"][idx])
                subj_type = d["subj_type"]
                obj_type = d["obj_type"]
                relation = d["relation"]
            
            processed.append({"len": seq_len, "tokens": tokens, "pos": pos, "ner": ner, "deprel": deprel, "position": positions, "s_type": subj_type, "o_type": obj_type, "relation": relation})
        return processed

def find_lowest_ancest(idx1, idx2, head):
    
    idx1_path = []
    idx2_path = []
    
    # subj-entity build path to root
    while idx1 != -1:
        idx1_path.append(idx1)
        idx1 = head[idx1] - 1
        
    # obj-entity build path to root, last idx2 = anc_idx
    while idx2 not in idx1_path:
        idx2_path.append(idx2)
        idx2 = head[idx2] - 1
        
    idx2_path.reverse()
    seq_idx = idx1_path + idx2_path
    
    return idx2, seq_idx
    
def get_positions(ss, se, os, oe, s_type, o_type, length):
    """ Get subj/obj position sequence. """
    if ss < os:
        return [1]*ss + [s_type]*(se - ss + 1) + [1]*(os - se - 1) + [o_type]*(oe - os + 1) + [1]*(length - oe - 1)
    else:
        return [1]*os + [o_type]*(oe - os + 1) + [1]*(ss - oe - 1) + [s_type]*(se - ss + 1) + [1]*(length - se - 1) 
        
if __name__ == "__main__":
    path = "/work/tacred/data/json/dev.json"
    
    relation = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
    
    entity = {"X": 0, "O": 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}
    
    dataloader = DataLoader(path, list(entity.keys()), list(relation.keys()), {"lower": True})
    data = dataloader.data
    
    for t in range(20):
        print(data[t]["len"])
        print(data[t]["tokens"])
        print(data[t]["pos"])
        print(data[t]["ner"])
        print(data[t]["deprel"])
        print(data[t]["position"])
        print(data[t]["s_type"])
        print(data[t]["o_type"])
        print(data[t]["relation"])