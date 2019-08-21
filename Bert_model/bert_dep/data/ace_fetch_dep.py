import sys
sys.path.append("/work/relation_extraction/Bert_model/baseline/data")
sys.path.append("/work/multi_doc_analyzer/")

from ace05_set_reader import ACE05Reader
from stanfordcorenlp import StanfordCoreNLP
import json
from tqdm import tqdm

# user defined 
read_path = "/work/LDC2006T06/dataset/"
output_path = "/work/relation_extraction/Bert_model/bert_dep/data/"
Set = ["train", "test", "dev"]

nlp_en = StanfordCoreNLP('http://140.109.19.190', port=9000, lang='en')
eng_reader = ACE05Reader('en', nlp_en)

for each_set in Set:
    data = {}
    doc_dicts = eng_reader.read(read_path + each_set + "/")
    file_dict = {}
    tqdm_key = tqdm(doc_dicts.keys())
    for file in tqdm_key:
        mydoc = doc_dicts[file]
        dep_list = []
        head_list = []
        for sen_idx in range(len(mydoc.sentences)):
            dep_sen_list = []
            head_sen_list = []
            for tok_idx in range(len(mydoc.sentences[sen_idx].tokens)):
                dep_sen_list.append(mydoc.sentences[sen_idx].tokens[tok_idx].dep_type)
                head_sen_list.append(mydoc.sentences[sen_idx].tokens[tok_idx].dep_head)
            dep_list.append(dep_sen_list)
            head_list.append(head_sen_list)
        file_dict["dep_type"] = dep_list
        file_dict["dep_head"] = head_list
        data[file] = file_dict
    with open(output_path + "ace_" + each_set + "_dep.json", 'w') as f:
        json.dump(data, f)