import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to input data')
parser.add_argument('--dir', type=str, help='path to input data')
args = parser.parse_args()

assert args.path is not None or args.dir is not None
assert args.path is None or args.dir is None

if args.dir:
    paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.csv')]
elif args.path:
    paths = [args.path]

for path in paths:
    print(f"Path = {path}")
    df = pd.read_csv(path)
    num_ones = len(df[df["score"] == 1])
    num_point5 = len(df[df["score"] == 0.5])
    pct = (num_ones + num_point5) / len(df) * 100
    print(f"  Soft win rate: {pct:.2f}")
