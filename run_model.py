import sys
import csv

import torch

import readwrite as rw
from bilinear import WordVectors, MarginalLogLinear, ConditionalSoftmax, ConditionalLogLinear, ConditionalLogBilinear

def main(model_filename, data_filename):
    model = torch.load(model_filename)
    groups = list(rw.read_groups(data_filename))
    scores = model(groups)
    writer = csv.writer(sys.stdout)
    for group, score in zip(groups, scores):
        writer.writerow(list(group) + [-score.item()])

if __name__ == '__main__':
    main(*sys.argv[1:])
    

    
    
