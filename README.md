# 📮 PostMark

Official repository for **"PostMark: A Robust Blackbox Watermark for Large Language Models"** 🌊

PostMark is a post-hoc watermarking method that operates without access to model logits, enabling it to watermark outputs from blackbox LLMs such as GPT-4. This repository provides the necessary annotations, outputs, and scripts to replicate the results reported in the accompanying paper. The code for running PostMark is currently being cleaned up and will be uploaded by mid-July 2024 at the latest.

## 🧭 Navigating the repo

```
.
└── PostMark/
    ├── annotations/
    │   ├── auto
    │   └── human/
    │       ├── pairwise
    │       └── spot
    ├── outputs/
    │   ├── c4
    │   ├── factscore
    │   ├── lfqa
    │   └── opengen
    └── postmark
```

- **`annotations/`**
    - `annotations/auto` contains all automatic annotations using GPT-4-Turbo as the judge for a pairwise comparison task, corresponding to Table 4.
    - `annotations/human/pairwise` contains human annotations for the pairwise comparison task (Section 4.2, Q3), and `annotations/human/spot` contains annotations for the watermark word identification task (Section 4.2, Q4).

- **`outputs/`**
    - Each file in the `outputs/{dataset}` directory follows the naming convention `{base-llm}_{watermarking_method}.jsonl`. Within these files, `text1` is the original output by the underlying LLM, `text2` is the watermarked output, and `text3` is the paraphrased watermarked output.

- **`postmark/`**
    - This directory contains code required to run PostMark.

## 💧 Running PostMark

### 🧩 Prerequisites 

1. Install requirements.
2. `python3 -m spacy download en_core_web_sm`.
3. Download all files in [this link](https://drive.google.com/drive/folders/1Rdpqbtvy2s91ZrcgqDy6CrTCb9dZBQBf?usp=sharing), place them in the directory.
4. Put your openai key in `openai_key.txt`.
5. If you want to use `llama-3-70b-chat`, put your together.ai key in `together_key.txt`, because currently PostMark uses the together API for accessing this model.

### 🧩 Watermarking

The command below will run PostMark with GPT-4 as the base LLM, OpenAI text-embedding-3-large as the embedder, and GPT-4o as the inserter on the OpenGen dataset with `r` set to 0.12 (corresponding to PostMark@12 in the paper). The paraphraser is GPT-3.5-Turbo. Please see `postmark/watermark.py` for more detailed descriptions of each argument.

```
python3 postmark/watermark.py \
    --dataset opengen \
    --output_path test.jsonl \
    --llm gpt-4 \
    --embedder openai \
    --inserter gpt-4o \
    --ratio 0.12 \
    --iterate v2 \
    --para \
    --paraphraser gpt-3.5-turbo \
    --n 5
```

### 🧩 Detection

The command below will compute word presence scores and print target TPR at 1% FPR. Please see `postmark/detect.py` for more detailed descriptions of each argument.

```
python3 postmark/detect.py \
    --input_path test.jsonl \
    --thresh 0.7 \
    --output_path test_with_scores.jsonl \
    --n 5
```

If `test_with_scores.jsonl` already exists and you just want to print the TPR numbers again, simply run the following:

```
python3 postmark/detect.py \
    --input_path test_with_scores.jsonl \
    --thresh 0.7 \
    --n 5
```

## 🔢 Replicating numbers reported in the paper

First, install required packages by running `pip3 install -r requirements.txt`.

### TPR numbers in Tables 1, 2, 5

To replicate the TPR numbers reported in Tables 1, 2, 5, you can use `print_tpr.py`, a script that prints TPR @ 1% FPR before and after paraphrasing attacks.

**Option 1**: Print TPR numbers for one single file. Example:

`python3 print_tpr.py --path outputs/opengen/gpt-4_postmark-12.jsonl`

**Option 2**: Print TPR numbers for an entire dataset. Example:

`python3 print_tpr.py --dir outputs/opengen`

### Soft win rates in Tables 4, 5 (automatic evaluation with GPT-4-Turbo)

To replicate the soft win rates reported in Tables 4 and 5, you can use `parse_auto_annots.py`:

**Option 1**: Print the soft win rate for one single file. Example:

`python3 parse_auto_annots.py --path annotations/auto/gpt-4_postmark-12.csv`

**Option 2**: Print the soft win rate for the entire directory. Example:

`python3 parse_auto_annots.py --dir annotations/auto`

### Section 4.2 (human evaluation)

To replicate numbers reported in Section 4.2 (Q3 and Q4), run `python3 parse_human_annots.py`.
