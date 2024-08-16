import json
import os
import numpy as np
import tiktoken
import spacy
import torch
from torch.nn.functional import cosine_similarity

from models import Paragram

nlp = spacy.load("en_core_web_sm")
encoding = tiktoken.get_encoding('cl100k_base')
paragram = Paragram(ratio=0.1)

with open("prompts/paraphrase_sent_init.txt") as f:
    paraphrase_template_init = f.read()
with open("prompts/paraphrase_sent.txt") as f:
    paraphrase_template = f.read()


def paraphrase(text, paraphraser, sent=True):
    doc = nlp(text)
    sentences = [str(s) for s in doc.sents]
    pp_sentences = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            prompt = paraphrase_template_init.format(sentence)
            pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
            pp_sentences.append(pp_sentence)
            continue
        context = ' '.join(sentences[:i])
        prompt = paraphrase_template.format(' '.join(pp_sentences), sentence)
        pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
        pp_sentences.append(pp_sentence)
    pp_text = ' '.join(pp_sentences)
    return pp_text


def get_similarities(list_words, text_words):
    list_words_embs = paragram.get_embeddings(list_words)
    text_words_embs = paragram.get_embeddings(text_words)
    sims = []
    for list_word_emb in list_words_embs:
        sim = cosine_similarity(text_words_embs, list_word_emb.unsqueeze(0))
        if sim.shape[0] != len(text_words):
            assert text_words_embs.shape[0] != len(text_words), f"{sim.shape[0]} != {len(text_words)}"
        sims.append(sim)
    sims = torch.stack(sims, dim=0).cpu()
    topk_scores, topk_indices = torch.topk(sims, 1, dim=1)
    topk_words = [[text_words[i] for i in indices] for indices in topk_indices]
    return topk_words, topk_scores


def compute_presence(text, words, threshold=0.7):
    text_words = [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]
    topk_words, topk_scores = get_similarities(words, text_words)
    present = 0
    for w, s in zip(words, topk_scores):
        if w.lower() in text_words or s[0] >= threshold:
            present += 1
    presence = present / len(words)
    return presence


def print_tpr_target(fpr, tpr, target_fpr=0.01):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    if target_fpr < fpr[0]:
        print(f"TPR at {target_fpr*100}% FPR: {tpr[0] * 100:5.1f}% (target too low)")
        return
    
    if target_fpr > fpr[-1]:
        print(f"TPR at {target_fpr*100}% FPR: {tpr[-1] * 100:5.1f}% (target too high)")
        return
    
    idx = np.searchsorted(fpr, target_fpr, side='right')
    
    if fpr[idx-1] == target_fpr:
        tpr_value = tpr[idx-1]
    else:
        tpr_value = tpr[idx-1] + (target_fpr - fpr[idx-1]) * (tpr[idx] - tpr[idx-1]) / (fpr[idx] - fpr[idx-1])
    
    print(f"TPR at {target_fpr*100}% FPR: {tpr_value * 100:5.1f}%")
