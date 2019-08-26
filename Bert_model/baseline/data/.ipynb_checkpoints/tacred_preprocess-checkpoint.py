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
            
        self.e_type2idx = entity_type
        self.r_type2idx = relation_type
        self.raw_data = raw_data
        data = self.preprocess(raw_data, opt)
        self.data = data

    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []

        tqdm_data = tqdm(data)
        if opt["lower"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        for d in tqdm_data:
            
            bert_tokenize(tokenizer, d, opt)
            tokens = list(d["token"])
            seq_len = len(tokens) + 2
    
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
         
            pos = d['stanford_pos']
            ner = d['stanford_ner']
            deprel = d['stanford_deprel']
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            
            positions = get_positions(d['subj_start']+1, d['subj_end']+1, d['obj_start']+1, d['obj_end']+1, self.e_type2idx[d["subj_type"]], self.e_type2idx[d["obj_type"]], seq_len)
            subj_type = d['subj_type']
            obj_type = d['obj_type']
            relation = self.r_type2idx[d['relation']]
            processed.append({"len": seq_len, "tokens": tokens, "pos": pos, "ner": ner, "deprel": deprel, "head": head, "position": positions, "s_type": subj_type, "o_type": obj_type, "relation": relation})
        return processed

def get_positions(ss, se, os, oe, s_type, o_type, length):
    """ Get subj/obj position sequence. """
    if ss < os:
        return [1]*ss + [s_type]*(se - ss + 1) + [1]*(os - se - 1) + [o_type]*(oe - os + 1) + [1]*(length - oe - 1)
    else:
        return [1]*os + [o_type]*(oe - os + 1) + [1]*(ss - oe - 1) + [s_type]*(se - ss + 1) + [1]*(length - se - 1) 
        
if __name__ == "__main__":
    path = "/work/tacred/data/json/test.json"
    
    relation = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
    
    entity = {"X": 0, "O": 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}
    
    dataloader = DataLoader(path, entity, relation, {"lower": True})
    data = dataloader.data
    
    for t in range(20):
        if data[t]["len"] > 0:
            t = 996
            print(data[t]["len"])
            print(data[t]["tokens"])
            print(data[t]["pos"])
            print(data[t]["ner"])
            print(data[t]["deprel"])
            print(data[t]["head"])
            print(data[t]["position"])
            print(data[t]["s_type"])
            print(data[t]["o_type"])
            print(data[t]["relation"])