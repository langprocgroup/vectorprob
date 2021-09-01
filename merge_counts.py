import sys
from collections import Counter

import readwrite as rw

def main(*filenames):
    c = Counter()
    for filename in filenames:
        c += rw.read_counts(filename)
    rw.write_counts(sys.stdout, c.items())

if __name__ == '__main__':
    main(*sys.argv[1:])
