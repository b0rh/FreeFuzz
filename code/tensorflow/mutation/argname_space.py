import os
import json

import textdistance
import random

import numpy as np
from numpy import exp

class ArgNameSpace():
    sim = {}
    data = {}
    def __init__(self, def_file="mutation/api_defs.txt", vs_dir="output/argment_value_space/"):
        self.api_arg_with_value = {}
        self._load(def_file, vs_dir)

    def parse_def(self, s):
        s = s.strip()
        pos0 = s.find('(')
        api = s[:pos0]
        s = s[pos0 + 1:-1]
        args = s.split(',')
        argnames = []
        for a in args:
            a = a.strip()
            if len(a) == 0: continue
            if not a[0].isalpha(): continue
            if '=' in a:
                argnames.append(a.split('=')[0])
            else:
                argnames.append(a)
        return api, argnames

    def _load(self, def_file, vs_dir):
        with open(def_file, 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            api, args = self.parse_def(line)
            self.data[api] = args

        for api in self.data.keys():
            vs_file = os.path.join(vs_dir, api[3:] + ".txt")
            if not os.path.exists(vs_file):
                self.api_arg_with_value[api] = []
                continue
            with open(vs_file, "r") as fr:
                arg_with_value = fr.readlines()[0]
                arg_with_value = json.loads(arg_with_value)
            args = list(arg_with_value.keys())
            args = list(filter(lambda k: len(arg_with_value[k]) > 0, args))
            self.api_arg_with_value[api] = args

    def get_argname(self, api_name, index):
        return self.data[api_name][index]

    def get_argnames(self, api_name):
        if not api_name in self.data:
            return []
        return self.data[api_name]

    def has_api(self, api):
        return api in self.data

    def search_api_with_arg(self, arg_name):
        res = []
        for api, args in self.data.items():
            if arg_name in args and arg_name in self.api_arg_with_value[api]:
                res.append(api)
        return res

    def clean_def(self, api_name):
        args = self.get_argnames(api_name)
        return "%s(%s)" % (api_name, ','.join(args))

    @staticmethod
    def str_sim(s1, s2):
        return textdistance.levenshtein.normalized_similarity(s1, s2)

    def api_sim(self, api1, api2):
        if (api1, api2) in self.sim:
            return self.sim[(api1, api2)]
        else:
            def1 = self.clean_def(api1)
            def2 = self.clean_def(api2)
            s = ArgNameSpace.str_sim(def1, def2)
            self.sim[(api1, api2)] = self.sim[(api2, api1)] = s
            return s

    def find_sim(self, api_name, arg_name):
        """ Returns an API name sampled from the full API list based on similarities. """
        apis = self.search_api_with_arg(arg_name)
        if len(apis) == 0:
            return None
        sims = [self.api_sim(api_name, a) for a in apis]

        def softmax(x):
            """ Compute softmax values. """
            x = np.array(x)
            x = exp(x - np.max(x))
            return x / x.sum(axis=0)

        probs = softmax(sims)
        x = random.random()
        y = 0
        for i in range(len(probs)):
            y += probs[i]
            if x >= y:
                return apis[i]
        return None