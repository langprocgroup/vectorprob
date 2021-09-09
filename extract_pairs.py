import sys
import csv
import itertools
from collections import Counter

import conllu

flat = itertools.chain.from_iterable

def extract_dependencies(node):
    for child in node.children:
        yield node.token, child.token, child.token['deprel']
        yield from extract_dependencies(child)

def extract_pairs_from_files(hpos, dpos, rel, filenames):
    for filename in filenames:
        print("Extracting from %s ..." % filename, file=sys.stderr)
        with open(filename) as infile:
            parses = conllu.parse_tree_incr(infile)
            dependencies = flat(map(extract_dependencies, parses))
            for h, d, rel in dependencies:
                if rel == rel and h['upos'] == hpos and d['upos'] == dpos:
                    pair = h['form'].lower(), d['form'].lower()
                    yield pair    

def main(hpos, dpos, rel, *filenames):
    pairs = extract_pairs_from_files(hpos, dpos, rel, filenames)
    counts = Counter(pairs)
    print("N tokens = %d" % sum(counts.values()), file=sys.stderr)
    print("N types = %d" % len(counts), file=sys.stderr)
    if counts:
        writer = csv.writer(sys.stdout)
        for (h, d), c in counts.items():
            writer.writerow([d, h, c])
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
                
                
            
            
            
