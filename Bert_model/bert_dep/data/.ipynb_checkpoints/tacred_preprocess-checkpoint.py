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
    def __init__(self, filename, entity_type, relation_type, dep_type, dep_sz, opt=False):

        with open(filename) as infile:
            raw_data = json.load(infile)
            
        self.e_type2idx = entity_type
        self.r_type2idx = relation_type
        self.d_type2idx = dep_type
        self.dep_sz = dep_sz
        self.embedding = torch.nn.Embedding(num_embeddings=len(dep_type), embedding_dim=dep_sz*dep_sz, padding_idx=0)
        self.raw_data = raw_data
        data = self.preprocess(raw_data, opt)
        self.data = data

    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        
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
            anc_idx, seq_idx, subj_path, obj_path = self.find_lowest_ancest(d["subj_start"], d["obj_start"], d["stanford_deprel"], d["stanford_head"])
            
            seq_len = len(d["token"]) + 2
#             print(seq_idx)
            positions = get_positions(d['subj_start']+1, d['subj_end']+1, d['obj_start']+1, d['obj_end']+1, self.e_type2idx[d["subj_type"]], self.e_type2idx[d["obj_type"]], seq_len)
            
            tokens = list(d["token"])
            tokens.append(Token(text="[SEP]"))
            tokens.append(Token(text=str(d["token"][anc_idx])))
            
            subj_type = d["subj_type"]
            obj_type = d["obj_type"]
            relation = self.r_type2idx[d["relation"]] 
            pos = d["stanford_pos"].copy()
            ner = d['stanford_ner'].copy()
            deprel = d['stanford_deprel'].copy()
            head = [int(x) for x in d['stanford_head']]
            subj_path = seq_idx
            
            dep_position = []
            for i in range(seq_len):
                if i-1 not in seq_idx:
                    dep_position.append(1)
                else:
                    dep_position.append(self.d_type2idx[d["stanford_deprel"][i-1]])
#             for i,idx in enumerate(seq_idx):
#                 tokens.append(d["token"][idx])
#                 pos.append(d["stanford_pos"][idx])
#                 ner.append(d["stanford_ner"][idx])
#                 deprel.append(d["stanford_deprel"][idx])
            
            processed.append({"len": seq_len, "tokens": tokens, "pos": pos, "ner": ner, "deprel": deprel, "position": positions, "s_type": subj_type, "o_type": obj_type, "subj_path": subj_path, "obj_path": obj_path, "dep_pos": dep_position, "relation": relation})
        return processed

    def find_lowest_ancest(self, idx1, idx2, dep, head):

        idx1_path = []
        idx2_path = []
        init1 = []
        init2 = []
        
        # subj-entity build path to root
        while idx1 != -1:
            idx1_path.append(idx1)
            idx1 = head[idx1] - 1

        # obj-entity build path to root, last idx2 = anc_idx
        while idx2 not in idx1_path:
            idx2_path.append(idx2)
            idx2 = head[idx2] - 1

#         # get subject dependency path vector
#         init1 = [1/self.dep_sz for i in range(self.dep_sz)]
#         for i in range(len(idx1_path)-1, -1, -1):
#             embeddings = self.embedding(torch.tensor(self.d_type2idx[dep[idx1_path[i]]]))
#             matrix = embeddings.detach().numpy().reshape(self.dep_sz, -1)
#             init1 = np.dot(init1, matrix)
        
#         # get object dependency path vector
#         init2 = [1/self.dep_sz for i in range(self.dep_sz)]
#         embeddings = self.embedding(torch.tensor(self.d_type2idx[dep[idx1_path[-1]]]))
#         matrix = embeddings.detach().numpy().reshape(self.dep_sz, -1)
#         init2 = np.dot(init2, matrix)
#         for i in range(len(idx2_path)-1, -1, -1):
#             embeddings = self.embedding(torch.tensor(self.d_type2idx[dep[idx2_path[i]]]))
#             matrix = embeddings.detach().numpy().reshape(self.dep_sz, -1)
#             init2 = np.dot(init2, matrix)
        
        idx2_path.reverse()
        seq_idx = idx1_path + idx2_path

        return idx2, seq_idx, list(init1), list(init2)
    
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
    
    dep = {"PAD": 0, "UNK": 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}
    
    dataloader = DataLoader(path, entity, relation, dep, 10, {"lower": True})
    data = dataloader.data
    
    for t in range(20):
        if data[t]["len"] > 0:
            print(t)
            print(data[t]["len"])
            print(data[t]["tokens"])
            print(data[t]["pos"])
            print(data[t]["ner"])
            print(data[t]["deprel"])
            print(data[t]["position"])
            print(data[t]["s_type"])
            print(data[t]["o_type"])
            print(data[t]["subj_path"])
            print(data[t]["obj_path"])
            print(data[t]["dep_pos"])
            print(data[t]["relation"])