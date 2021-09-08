import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import choice
from mutation.read_db import get_all_signature
from termcolor import colored

from mutation.random_utils import *
from mutation.TFAPI import TFAPI
from mutation.utils import code_to_file, run_code


class API_VS(object):
    def __init__(self, api_name, db_path, selection_mode="random"):
        self.api_name = api_name
        self.db_path = db_path
        self.db_file = os.path.join(db_path, api_name.replace("tf.", "") + '.txt')
        self.selection_mode = selection_mode
        self.index = 0
        self.api_cnt = 0
        self.api_list = []
        self.api_counts = []
        self._read_records()

    def _read_records(self):
        self.records = get_all_signature(self.db_file)
        self.record_cnt = len(self.records)

    def sample_records(self, count):
        self.records = choice(self.records, count)

    def build_api_from_records(self,
                               dump_code=True,
                               output_dir="tempfile",
                               dump_import=True,
                               call_api=True,
                               run_dumped_code=False,
                               verbose=False):
        # Build TFAPI lists from records
        count, success_count = 0, 0
        api_list: list[TFAPI] = []
        api_counts: list[int] = []
        for record in self.records:
            if verbose:
                print(colored(record, "yellow"))

            api = TFAPI(self.db_file, record, self.api_name)
            success = False
            code = api.to_code()
            if verbose:
                print(code)
            if dump_code:
                code_to_file(code, "%s-%d" % (self.api_name, count), output_dir, dump_import)
            if call_api:
                success = api.call()
            if run_dumped_code:
                success = run_code(code, "%s-%d" % (self.api_name, count), output_dir) == 0
                if verbose:
                    if success:
                        print(colored("[passed] run_code", "green"))
                    else:
                        print(colored("[failed] run_code", "red"))


            count += 1
            success_count += success
            api_list.append(api)
            api_counts.append(1)

        self.api_list = api_list
        self.api_counts = api_counts
        self.api_cnt = count
        return count, success_count

    def next(self):
        if self.selection_mode == "random":
            if self.api_cnt <= 0: return None
            if self.api_cnt <= 1: self.index = 0
            else:
                self.index = random.randint(0, self.api_cnt - 1)
            return self.api_list[self.index]
        elif self.selection_mode == "sequential":
            self.index += 1
            if self.index > self.api_cnt:
                self.index = 1
            return self.api_list[self.index - 1]