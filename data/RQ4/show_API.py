import sys
from os.path import split

def readtxt(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
        return len(lines)

if  __name__ == "__main__":
    path_to_file = sys.argv[1]
    filename = split(path_to_file)[1].replace(".txt", "")
    print("  %s: %d APIs covered" % (filename, readtxt(path_to_file)))
