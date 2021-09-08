import sys
import json

def sum_lines(dict_cov):
    res = 0
    for key in dict_cov.keys():
        res += len(dict_cov[key]) # add up the number of covered lines
    return res

def readjson(filename):
    with open(filename, "r") as fin:
        return json.load(fin)

def readtxt(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
        return len(lines)

if  __name__ == "__main__":
    path_to_file = sys.argv[1]
    temp = readjson(path_to_file)
    filename = path_to_file.replace(".json", "")
    apicov_file = filename + ".txt"
    print("  %s: %d lines covered with %d APIs covered" % (filename, sum_lines(temp), readtxt(apicov_file)))
