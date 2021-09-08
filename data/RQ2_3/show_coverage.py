import sys
import json
from os.path import join

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
    for i in range(0, 1001, 100):
        temp = readjson(join(path_to_file, "%s.json" % (str(i))))
        print("  %s: %d" % (str(i), sum_lines(temp)))
