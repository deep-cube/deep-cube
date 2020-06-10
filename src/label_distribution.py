import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint


def plot(path):

    # list files
    files = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    jsonfiles = [f for f in files if f.split('.')[-1] == 'json']

    # load files
    d = defaultdict(int)
    for jsonfile in jsonfiles:
        print(f'processing {jsonfile}')
        with open(os.path.join(path, jsonfile)) as f:
            label = json.load(f)

            for k in label['moves']:
                d[label['moves'][k]] += 1
    # pprint(d)

    sortedtuples = sorted([(d[k], k) for k in d], reverse=True)
    vals, counts = zip(*sortedtuples)

    plt.bar(counts, vals, 2)
    plt.show()


if __name__ == "__main__":
    src_dir = sys.argv[1]
    plot(src_dir)
