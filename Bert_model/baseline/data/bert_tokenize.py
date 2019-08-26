from allennlp.data.tokenizers import Token

def bert_tokenize(tokenizer, d: dict, opt: dict={"lower": False}):
    counter = 0
    token = []
    pos = []
    ner = []
    head = []
    dep_rel = []
    ss, se = d["subj_start"], d["subj_end"]
    os, oe = d["obj_start"], d["obj_end"]
    idx_map = [0 for i in range(len(d["token"]))]
    
    # iter through all the current tokens
    for i,word in enumerate(d["token"]):
        # special term replace
        if word == "-LRB-" or word == "LSB" or word == "LCB":
            word = "("
        elif word == "-RRB-" or word == "-RSB-" or word == "RCB":
            word = ")"
        
        # get new token from original one
        tok_word = tokenizer.tokenize(word)
        idx_map[i] = len(token)
        
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
            
    # to new head idx
    for i in range(len(head)):
        if head[i] != 0: 
            head[i] = idx_map[head[i] - 1] + 1
        
    # re-assign
    d["token"] = token.copy()
    d["stanford_pos"] = pos.copy()
    d["stanford_ner"] = ner.copy()
    d["stanford_head"] = head.copy()
    d["stanford_deprel"] = dep_rel.copy()
    d["subj_start"], d["subj_end"] = ss, se
    d["obj_start"], d["obj_end"] = os, oe