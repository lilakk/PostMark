import pandas as pd
import ast
import math
import json

df_hard = pd.read_csv("annotations/human/pairwise/postmark-12.csv")
df_soft = pd.read_csv("annotations/human/pairwise/postmark-6.csv")

def compute_score(row, aspect='overall'):
    if row[aspect] == 'A':
        if row['text2'] == 'A':
            return 1
        elif row['text1'] == 'A':
            return 0
    elif row[aspect] == 'B':
        if row['text2'] == 'B':
            return 1
        elif row['text1'] == 'B':
            return 0
    elif row[aspect] == 'Tie':
        return 0.5
    return None

aspects = ["overall", "relevance", "coherence", "interestingness"]
print("[Section 4.2 | Q3] Results for the pairwise comparison task:")
for aspect in aspects:
    df_hard[f'score_{aspect}'] = df_hard.apply(lambda row: compute_score(row, aspect=aspect), axis=1)
    df_soft[f'score_{aspect}'] = df_soft.apply(lambda row: compute_score(row, aspect=aspect), axis=1)
    print(f"  Aspect: {aspect}")
    hard_score = (len(df_hard[df_hard[f'score_{aspect}'] == 1]) + len(df_hard[df_hard[f'score_{aspect}'] == 0.5])) / len(df_hard) * 100
    soft_score = (len(df_soft[df_soft[f'score_{aspect}'] == 1]) + len(df_soft[df_soft[f'score_{aspect}'] == 0.5])) / len(df_soft) * 100
    print(f"    PostMark@12: {hard_score:5.1f}")
    print(f"    PostMark@6: {soft_score:5.1f}")

def determine_identity(custom_id):
    custom_id = str(custom_id)
    if custom_id.endswith('1'):
        return 'text1'
    elif custom_id.endswith('2'):
        return 'text2'
    else:
        return 'other'

def find_list2(text2, data):
    for dd in data:
        if dd['text2'] == text2:
            return dd['list2']
    return None

def print_spot_stats(all_text1, all_text2, data):
    print("[Section 4.2 | Q4] Results for word-spotting task:")
    total_marked_text1 = []
    for i, row in all_text1.iterrows():
        if type(row['label']) == str:
            labels = ast.literal_eval(row['label'])
            annot_list = [x["text"] for x in labels]
        elif math.isnan(row['label']):
            annot_list = []
        total_marked_text1.extend(annot_list)
    print(f"  avg # words marked in text1: {len(total_marked_text1) / len(all_text1):.2f}")
    total_marked_text2 = []
    all_precision = []
    all_recall = []
    all_f1 = []
    for i, row in all_text2.iterrows():
        if type(row['label']) == str:
            labels = ast.literal_eval(row['label'])
            annot_list = [x["text"] for x in labels]
        elif math.isnan(row['label']):
            annot_list = []
        total_marked_text2.extend(annot_list)
        annot_list = [x.lower().strip() for x in annot_list]
        actual_list = find_list2(row['text'], data)
        actual_list = [x.lower().strip() for x in actual_list]
        tp = len(set(annot_list) & set(actual_list))
        fp = len(set(annot_list) - set(actual_list))
        fn = len(set(actual_list) - set(annot_list))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    print(f"  avg # words marked in text2: {len(total_marked_text2) / len(all_text2):.2f}")
    print(f"  precision: {sum(all_precision) / len(all_precision):.2f}")
    print(f"  recall: {sum(all_recall) / len(all_recall):.2f}")
    print(f"  f1: {sum(all_f1) / len(all_f1):.2f}")

data_path = "outputs/opengen/gpt-4_postmark-12.jsonl"
data = [json.loads(line) for line in open(data_path, 'r')][20:40]
df = pd.read_csv("annotations/human/spot/postmark-12.csv")
df['identity'] = df['custom_id'].apply(determine_identity)
all_text1 = df[df['identity'] == 'text1']
all_text2 = df[df['identity'] == 'text2']
print_spot_stats(all_text1, all_text2, data)
