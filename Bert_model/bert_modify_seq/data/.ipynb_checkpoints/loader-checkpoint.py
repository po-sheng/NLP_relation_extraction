"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

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
        
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            
            tokens[ss:se+1] = ["[unused" + str(e_type2idx(d['subj_type'])) + "]"] * (se-ss+1)
            tokens[os:oe+1] = ["[unused" + str(e_type2idx(d['obj_type']) + len(entity_type)) + "]"] * (oe-os+1)
#             do bert tokenize
#             todo

            pos = d['stanford_pos']
            ner = d['stanford_ner']
            deprel = d['stanford_deprel']
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = d['subj_type']
            obj_type = d['obj_type']
            relation = r_type2idx[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

if __name__ == "__main__":
    path = "/work/tacred/data/json/dev.json"
    data = DataLoader(path, {"lower": True})
    