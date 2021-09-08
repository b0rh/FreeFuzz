
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from termcolor import colored

from mutation.random_utils import *
from mutation.TFAPI import TFAPI
from mutation.utils import code_to_file


def get_number_of_mutate_argument(total):
    if total == 0: return 0
    return random.randint(1, total)


def doTypeMutation(stat):
    if not stat["type_mutation"]: return False
    return random.random() < 0.33


def doDBValueMutation(stat):
    if not stat["db_mutation"]: return False
    return random.random() < 0.25

def doRandomValueMutation(stat):
    if not stat["random_mutation"]: return False
    return True

if __name__ == "__main__":

    record_seed = {"filters": {"Label": "raw", "value": "4"}, "kernel_size": {"Label": "raw", "value": "2"}, "input_signature": [{"Label": "tensor", "dtype": "float64", "shape": [4, 4, 4, 4]}], "output_signature": {"Label": "tensor", "dtype": "float32", "shape": [4, 3, 3, 4]}}
    api_name = "tf.keras.layers.Conv2D"
    api_seed: TFAPI = TFAPI(record=record_seed, api_name=api_name)
    mutation_count = 20

    output_dir = "output/"
    
    stat = {
        "type_mutation": True,
        "random_mutation": True,
        "db_mutation": True,
    }
    api: TFAPI = TFAPI(record=api_seed.record, api_name=api_seed.api_name)

    arg_cnt = api.get_number_of_arguments()
    mutate_arg_cnt = get_number_of_mutate_argument(arg_cnt)
    mutate_arg_names = api.random_arg_names(mutate_arg_cnt)

    if doTypeMutation(stat):
        api.mutate_type(mutate_arg_names)

    mutated_arg_names = []

    if doDBValueMutation(stat):
        mutated_arg_names = api.mutate_value_db(mutate_arg_names)

    if doRandomValueMutation(stat):
        rest_args = list(set(mutate_arg_names) - set(mutated_arg_names))
        api.mutate_value_random(rest_args)
    
    code_origin = api_seed.to_code()
    
    filename_origin = "example_Conv2D"
    code_to_file(code_origin, filename_origin, output_dir)

    code = api.to_code()
    
    filename_mutation = "example_Conv2D_mutation"
    code_to_file(code, filename_mutation, output_dir)
