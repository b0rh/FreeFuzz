import sys
import json
from os.path import split

def sum_lines(dict_cov):
    res = 0
    for key in dict_cov.keys():
        res += len(dict_cov[key]) # add up the number of covered lines
    return res

def readjson(filename):
    with open(filename, "r") as fin:
        return json.load(fin)

if  __name__ == "__main__":
    path_to_file = sys.argv[1]
    temp = readjson(path_to_file)
    filename = split(path_to_file)[1].replace(".json", "")
    print("  %s: %d lines covered" % (filename, sum_lines(temp)))
