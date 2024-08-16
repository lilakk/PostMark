import argparse
import json
import tiktoken
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score
from utils import compute_presence, print_tpr_target

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, help='Path to the JSONL data to be evaluated.')
parser.add_argument('--thresh', type=float, help='Cosine similarity threshold to use for computing word presence.', default=0.7)

parser.add_argument('--output_path', type=str, help='(Optional) JSONL path to save the scores along with the input data.')
parser.add_argument('--n', type=int, help='(Optional) Number of examples to evaluate.')

args = parser.parse_args()

encoding = tiktoken.get_encoding('cl100k_base')

if args.n:
    data = [json.loads(line) for line in open(args.input_path, 'r')][:args.n]
else:
    data = [json.loads(line) for line in open(args.input_path, 'r')]

para = "text3" in data[0]  # whether the input data contains paraphrased text

text1s = [dd['text1'] for dd in data]
list1s = [dd['list1'] for dd in data]
score1s = []
text2s = [dd['text2'] for dd in data]
list2s = [dd['list2'] for dd in data]
score2s = []
if para:
    text3s = [dd['text3'] for dd in data]
    list3s = [dd['list3'] for dd in data]
    score3s = []
if "score1" in data[0]:
    score1s = [dd['score1'] for dd in data]
    score2s = [dd['score2'] for dd in data]
    if para:
        score3s = [dd['score3'] for dd in data]

if not score1s:
    for i in tqdm(range(len(data)), total=len(data), desc="Obtaining presence scores"):
        score1s.append(compute_presence(text1s[i], list1s[i], threshold=args.thresh))
        score2s.append(compute_presence(text2s[i], list2s[i], threshold=args.thresh))
        if para:
            score3s.append(compute_presence(text3s[i], list3s[i], threshold=args.thresh))

# save scores if output path is provided
if args.output_path:
    with open(args.output_path, 'w') as fout:
        for i in range(len(data)):
            record = {
                "text1": text1s[i],
                "list1": list1s[i],
                "score1": score1s[i],
                "text2": text2s[i],
                "list2": list2s[i],
                "score2": score2s[i]
            }
            if para:
                record.update({
                    "text3": text3s[i],
                    "list3": list3s[i],
                    "score3": score3s[i]
                })
            fout.write(json.dumps(record) + "\n")

labels = [0] * len(data) + [1] * len(data)

scores = score1s + score2s
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
print_tpr_target(fpr, tpr, target_fpr=0.01)
print(f"AUC: {roc_auc}")

if para:
    print("\nAfter paraphrase attack:")
    scores = score1s + score3s
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print_tpr_target(fpr, tpr, target_fpr=0.01)
    print(f"AUC: {roc_auc}")
