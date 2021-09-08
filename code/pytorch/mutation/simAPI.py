import textdistance
import re
import numpy as np
import json

API_def = {}
API_args = {}

def string_similar(s1, s2):
    return textdistance.levenshtein.normalized_similarity(s1, s2)


def loadAPIs():
    global API_def, API_args
    with open("APIdef.txt", "r") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            API_name = line.split("(")[0]
            API_args_match = re.search("\((.*)\)", line)
            API_args_text = API_args_match.group(1)
            if API_name not in API_def.keys():
                API_def[API_name] = line
                API_args[API_name] = API_args_text

def query_argname(argname):
    '''
    Return a list of APIs with the exact argname
    '''
    APIs = []
    for key in API_args.keys(): # key is API_name
        if argname in API_args[key]:
            argVSfile = key.replace("torch.", "") + ".txt"
            try:
                with open("argVS/" + argVSfile, "r") as fin:
                    d = json.load(fin)
                    if argname in d.keys():
                        if len(d[argname]) != 0:
                            APIs.append(key)
            except Exception as e:
                continue
    return APIs

def mean_norm(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def similarAPI(API, argname):
    '''
    Return a list of similar APIs (with the same argname) and their similarities
    '''
    API_with_same_argname = query_argname(argname)
    if len(API_with_same_argname) == 0:
        return [] , []
    probs = []
    original_def = API_def[API]
    for item in API_with_same_argname:
        to_compare = API_def[item]
        probs.append(string_similar(original_def, to_compare))
    prob_norm2 = softmax(probs)
    return API_with_same_argname, prob_norm2


loadAPIs()