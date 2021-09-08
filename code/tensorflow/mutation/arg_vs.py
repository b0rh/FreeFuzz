import json
import os
from numpy.random import choice


class ArgmentValueSpace(object):
    def __init__(self, vs_dir="output/argment_value_space/"):
        self.vs_dir = vs_dir
        self.vs_data = {}

    def sample_arg_signature_from_vs(self, api_name, arg_name):
        if not api_name in self.vs_data:
            fn = os.path.join(self.vs_dir, "%s.txt" % (api_name.replace('tf.','')))
            if not os.path.exists(fn):
                return None
            with open(fn, 'r') as fr:
                data = fr.readlines()[0]
            self.vs_data[api_name] = data
        else:
            data = self.vs_data[api_name]

        d = json.loads(data)
        if not arg_name in d: return None
        vs = d[arg_name]
        if len(vs) == 0:
            return None
        return choice(vs, 1)