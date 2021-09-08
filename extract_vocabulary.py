import sys
from collections import Counter

import readwrite as rw

def marginalize(groupcounts, column):
    c = Counter()
    for group, count in groupcounts:
        c[group[column]] += int(count)
    return c

def main(filename, cutoff):
    cutoff = int(cutoff)
    counts = rw.read_counts(filename)
    marg = marginalize(counts.items(), 0)
    sorted_words = sorted([(count, word) for word, count in marg.items()])
    keep = [word for count, word in sorted_words[-cutoff:]]
    for word in keep:
        print(word)
    
if __name__ == '__main__':
    main(*sys.argv[1:])
    
