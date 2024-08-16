import os
import tqdm
import torch
import json
import argparse
import time
import spacy
import random
from utils import paraphrase
from models import Watermarker, LLM

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='Which dataset to use.', choices=["opengen", "factscore", "lfqa"], default="opengen")
parser.add_argument('--output_path', type=str, help='JSONL path to save output data.')

parser.add_argument('--llm', type=str, help='The base LLM to watermark the outputs of.', choices=["gpt-4", "llama-3-8b", "llama-3-8b-chat", "mistral-7b-inst"], default="gpt-4")
parser.add_argument('--embedder', type=str, help='The embedding model for selecting watermark words.', choices=["openai", "nomic"], default="openai")
parser.add_argument('--inserter', type=str, help='The LLM for inserting watermark words.', choices=["gpt-4o", "llama-3-70b-chat"], default="gpt-4o")
parser.add_argument('--ratio', type=float, help='Determines how many watermark words to insert.', default=0.12)
parser.add_argument('--iterate', type=str, help='Whether to use iterative insertion. Defaults to v2.', default="v2")

parser.add_argument('--para', action='store_true', help='Whether to generate paraphrases of watermarked text.')
parser.add_argument('--paraphraser', type=str, help='Which paraphraser model to use.', choices=["gpt-3.5-turbo", "gpt-4o"], default="gpt-3.5-turbo")

parser.add_argument('--n', type=int, help='(Optional) Number of dataset instances to iterate over.')
parser.add_argument('--cache_text1', type=str, help='(Optional) Path to a jsonl file that contains existing text1s to be reused')
args = parser.parse_args()
print(args)

if args.para:
    paraphraser = LLM(args.paraphraser)

def create_generation_record(res, text3=None, list3=None):
    record = res
    if args.para:
        record.update({
            "text3": text3, 
            "list3": list3
        })
    return record

def append_to_output_file(output_path, generation_record):
    with open(output_path, 'a') as fout:
        fout.write(json.dumps(generation_record) + "\n")

if args.dataset == "opengen":
    input_path = "data/opengen.jsonl"
    template = "Generate a continuation for the following text in 200-250 words, only include the continuation in your response.\n\n{}"
elif args.dataset == "lfqa":
    input_path = "data/lfqa.jsonl"
    template = "Answer the following question in 200-250 words.\n\n{}"
elif args.dataset == "factscore":
    input_path = "data/factscore.jsonl"
    template = "Tell me a bio of {}."
else:
    raise NotImplementedError

output_path = args.output_path

watermarker = Watermarker(args.llm,
                            args.embedder,
                            args.inserter,
                            ratio=args.ratio,
                            iterate=args.iterate)

if args.n:
    input_data = [json.loads(line) for line in open(input_path, 'r')][:args.n]
else:
    input_data = [json.loads(line) for line in open(input_path, 'r')]
print(f"Loaded {len(input_data)} examples for {args.dataset}.")

generations = []
if os.path.exists(output_path):
    generations = [json.loads(line) for line in open(output_path, 'r')]
    print(f"Loaded {len(generations)} existing generations from {output_path}.")

text1s = None
if args.cache_text1:
    text1s = [json.loads(line) for line in open(args.cache_text1, 'r')]
    print(f"Loaded {len(text1s)} cached text1s from {args.cache_text1}.")

for idx, dd in tqdm.tqdm(enumerate(input_data)):
    if idx < len(generations):
        if args.para and "text3" not in generations[idx]:
            print(f"Data exists but no paraphrases, adding now.")
            if 'text2' in generations[idx] and "text3" not in generations[idx]:
                generations[idx]["text3"] = paraphrase(generations[idx]["text2"], paraphraser)
                generations[idx]["list3"] = watermarker.get_words(generations[idx]["text3"], k=args.k, filter_freq=args.filter_freq, use_kmeans=args.kmeans)['words']
            with open(output_path, 'w') as fout:
                for generation in generations:
                    fout.write(json.dumps(generation) + "\n")
        continue
    
    text1 = None
    if text1s is not None and idx < len(text1s):
        text1 = text1s[idx]['text1']
    else:
        if args.dataset == "factscore":
            prompt = template.format(dd["title"])
        else:
            prompt = template.format(dd["prefix"])
        text1 = watermarker.llm.generate(prompt, max_tokens=1500, temperature=1)
    res = watermarker.insert_watermark(text1, max_tokens=1500)
    if args.para:
        text3 = paraphrase(res['text2'], paraphraser)
        list3 = watermarker.get_words(text3)
    generation_record = create_generation_record(res, text3, list3)
    append_to_output_file(output_path, generation_record)
