import argparse
import json
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to input data')
parser.add_argument('--dir', type=str, help='path to input data')
parser.add_argument('--n', type=int, help='number of instances to score', default=500)
args = parser.parse_args()

assert args.path is not None or args.dir is not None
assert args.path is None or args.dir is None

if args.dir:
    paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.jsonl')]
elif args.path:
    paths = [args.path]

def print_tpr_target(scores, labels, target_fpr=0.01, pp=False):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    prefix = "TPR" if not pp else "PP TPR"
    if target_fpr < fpr[0]:
        print(f"  > {prefix} at {target_fpr*100}% FPR: {tpr[0] * 100:5.1f}% (target FPR too low)")
        return
    if target_fpr > fpr[-1]:
        print(f"  > {prefix} at {target_fpr*100}% FPR: {tpr[-1] * 100:5.1f}% (target FPR too high)")
        return
    idx = np.searchsorted(fpr, target_fpr, side='right')
    if fpr[idx-1] == target_fpr:
        tpr_value = tpr[idx-1]
    else:
        tpr_value = tpr[idx-1] + (target_fpr - fpr[idx-1]) * (tpr[idx] - tpr[idx-1]) / (fpr[idx] - fpr[idx-1])
    print(f"  > {prefix} at {target_fpr*100}% FPR: {tpr_value * 100:5.1f}%")

for path in paths:
    print(f"Path = {path}")
    if args.n:
        data = [json.loads(line) for line in open(path, 'r')][:args.n]
    else:
        data = [json.loads(line) for line in open(path, 'r')]
    labels = []
    scores = []
    pp_labels = []
    pp_scores = []
    for dd in data:
        if 'score1' in dd:
            score1 = dd['score1']
            score2 = dd['score2']
            score3 = dd['score3']
        elif 'detect1' in dd and 'score' in dd['detect1']:
            score1 = dd['detect1']['score']
            score2 = dd['detect2']['score']
            score3 = dd['detect3']['score']
        elif 'detect1' in dd and 'z_score' in dd['detect1']:
            score1 = dd['detect1']['z_score']
            score2 = dd['detect2']['z_score']
            score3 = dd['detect3']['z_score']
        elif 'detect1' in dd and 'p_value' in dd['detect1']:
            score1 = -dd['detect1']['p_value']
            score2 = -dd['detect2']['p_value']
            score3 = -dd['detect3']['p_value']
        labels.append(0)
        labels.append(1)
        scores.append(score1)
        scores.append(score2)
        pp_labels.append(0)
        pp_labels.append(1)
        pp_scores.append(score1)
        pp_scores.append(score3)
    print_tpr_target(scores, labels, pp=False)
    print_tpr_target(pp_scores, pp_labels, pp=True)
