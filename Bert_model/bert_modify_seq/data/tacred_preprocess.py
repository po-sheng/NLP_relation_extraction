"""
Data loader for TACRED json files.
"""

import json
import random
import torch
from tqdm import tqdm
import numpy as np
from pytorch_transformers.tokenization_bert import BertTokenizer
from allennlp.data.tokenizers import Token

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
        max_len = opt["len"]
        
        tqdm_data = tqdm(data)
        if opt["lower"]:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        for d in tqdm_data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            
            bert_tokenize(tokenizer, d, opt)
            tokens = list(d["token"])
            seq_len = len(tokens) + 2

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens[ss:se+1] = [Token(text="[unused" + str(e_type2idx[d['subj_type']]) + "]")] * (se-ss+1)
            tokens[os:oe+1] = [Token(text="[unused" + str(e_type2idx[d['obj_type']] + len(entity_type)) + "]")] * (oe-os+1)
            tokens.append(Token(text="[SEP]"))
            for i in range(ss, se+1):
                tokens.append(d["token"][i])
            tokens.append(Token(text="[SEP]"))
            for i in range(os, oe+1):
                tokens.append(d["token"][i])          
            pos = d['stanford_pos']
            ner = d['stanford_ner']
            deprel = d['stanford_deprel']
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            
            subj_positions = get_positions(d['subj_start']+1, d['subj_end']+1, seq_len, max_len)
            obj_positions = get_positions(d['obj_start']+1, d['obj_end']+1, seq_len, max_len)
            subj_type = d['subj_type']
            obj_type = d['obj_type']
            relation = r_type2idx[d['relation']]
            processed.append({"len": seq_len, "tokens": tokens, "pos": pos, "ner": ner, "deprel": deprel, "head": head, "s_position": subj_positions, "o_position": obj_positions, "s_type": subj_type, "o_type": obj_type, "relation": relation})
        
        return processed

def get_positions(start_idx, end_idx, length, max_len):
    """ Get subj/obj position sequence. """
    return list(range(start_idx, 0, -1)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1 + max_len, length-end_idx+max_len))

def bert_tokenize(tokenizer, d, opt):
    counter = 0
    token = []
    pos = []
    ner = []
    head = []
    dep_rel = []
    ss, se = d["subj_start"], d["subj_end"]
    os, oe = d["obj_start"], d["obj_end"]
    for i,word in enumerate(d["token"]):
        tok_word = tokenizer.tokenize(word)
        if i == d["subj_start"]:
            ss += counter
        if i == d["obj_start"]:
            os += counter        
        counter += len(tok_word) - 1
        if i == d["subj_end"]:
            se += counter
        if i == d["obj_end"]:
            oe += counter

        for sub_word in tok_word:
            token.append(Token(text=sub_word))
            pos.append(d["stanford_pos"][i])
            ner.append(d["stanford_ner"][i])
            head.append(d["stanford_head"][i])
            dep_rel.append(d["stanford_deprel"][i])
    d["token"] = token
    d["stanford_pos"] = pos
    d["stanford_ner"] = ner
    d["stanford_head"] = head
    d["stanford_deprel"] = dep_rel
    d["subj_start"], d["subj_end"] = ss, se
    d["obj_start"], d["obj_end"] = os, oe
        
if __name__ == "__main__":
    path = "/work/tacred/data/json/dev.json"

    relation = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
    
    entity = {'PERSON': 0, 'ORGANIZATION': 1, 'DATE': 2, 'NUMBER': 3, 'TITLE': 4, 'COUNTRY': 5, 'LOCATION': 6, 'CITY': 7, 'MISC': 8, 'STATE_OR_PROVINCE': 9, 'DURATION': 10, 'NATIONALITY': 11, 'CAUSE_OF_DEATH': 12, 'CRIMINAL_CHARGE': 13, 'RELIGION': 14, 'URL': 15, 'IDEOLOGY': 16}
    
    dataloader = DataLoader(path, list(entity.keys()), list(relation.keys()), {"lower": True, "len": 400})
    data = dataloader.data

    for t in range(20):
        print(data[t]["len"])
        print(data[t]["tokens"])
        print(data[t]["pos"])
        print(data[t]["ner"])
        print(data[t]["deprel"])
        print(data[t]["head"])
        print(data[t]["s_position"])
        print(data[t]["o_position"])
        print(data[t]["s_type"])
        print(data[t]["o_type"])
        print(data[t]["relation"])