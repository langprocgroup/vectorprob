import sys
import readwrite as rw

def main(filename):
    counts = rw.read_counts(filename)
    return sum(counts.values())

if __name__ == '__main__':
    print(main(*sys.argv[1:]))
