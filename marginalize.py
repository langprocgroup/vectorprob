import sys
import csv
from collections import Counter

import tqdm

def marginalize(lines, column):
    c = Counter()
    for *parts, count in tqdm.tqdm(lines):
        c[parts[column]] += int(count)
    return c

def main(filename, column):
    column = int(column)
    with open(filename) as infile:
        reader = csv.reader(infile)
        c = marginalize(reader, column)
    writer = csv.writer(sys.stdout)
    #writer.writerow(header)
    for item in c.items():
        writer.writerow(item)

if __name__ == '__main__':
    main(*sys.argv[1:])
        
        
