"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import sys
sys.path.append("/work/AGGCN_TACRED/utils")
from data.loader import DataLoader
from model.trainer import GCNTrainer
import torch_utils, scorer, constant, helper
from vocab import Vocab
import csv

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
index = 0
error = [[[] for j in range(len(id2label.keys()))] for i in range(len(id2label.keys()))]
for i, b in enumerate(batch_iter):
    preds, probs, label, token, sub_pos, obj_pos, _ = trainer.predict(b)
    for j in range(len(token)):
        idx = index
        index += 1
        error[label[j]][preds[j]].append({"idx":idx, "token":[i for i in token[j].tolist() if i != 0], "sub_pos":sub_pos[j].tolist(), "obj_pos":obj_pos[j].tolist(), "preds":int(preds[j]), "label":int(label[j])})
    predictions += preds
    all_probs += probs

csvfile = open("/work/relation_extraction/AGGCN_TACRED/analysis/error_test.csv", "w", newline='')
writer = csv.writer(csvfile)
writer.writerow(["sentence", "idx", "predict", "gold"])
for i in tqdm(range(len(error))):
    for j in range(len(error[i])):
        inside = 0
        for k in range(len(error[i][j])):
            inside = 1
            sentence = " ".join([vocab.id2word[g] for g in error[i][j][k]["token"]])
            sub = []
            obj = []
            for it,g in enumerate(error[i][j][k]["sub_pos"]):
                if g == 0 and it < len(error[i][j][k]["token"]):
                    sub.append(vocab.id2word[error[i][j][k]["token"][it]])
            for it,g in enumerate(error[i][j][k]["obj_pos"]):
                if g == 0 and it < len(error[i][j][k]["token"]):
                    obj.append(vocab.id2word[error[i][j][k]["token"][it]])
            predict = id2label[error[i][j][k]["preds"]]
            gold = id2label[error[i][j][k]["label"]]
            writer.writerow([sentence, idx, predict, gold])
        if inside == 1:
            writer.writerow("")
    
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

