import json

from numpy.random import choice


def get_single_signature(filename) -> dict:
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        record = json.loads(line)
        return record


def get_rand_single_signature(filename) -> dict:
    with open(filename, 'r') as f:
        lines = f.readlines()
    return json.loads(choice(lines))


def get_all_signature(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records