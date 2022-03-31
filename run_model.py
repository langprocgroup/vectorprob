import sys
import csv

import torch

import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogBilinear

def main(model_filename, data_filename):
    model = torch.load(model_filename)
    groups = rw.read_groups(data_filename)
    pairs = [(one, two) for one, two, *_ in groups]
    scores = model(pairs)
    writer = csv.writer(sys.stdout)
    for pair, score in zip(pairs, scores):
        writer.writerow(list(pair) + [-score.item()])

if __name__ == '__main__':
    main(*sys.argv[1:])
    
