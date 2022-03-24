from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_file_distribution(wav_files: List[str], labels_list=None) -> None:
    """
    Plot the distribution of the files for each category and per label

    Args:
        wav_files: The list of filenames that are used
        labels_list: The correct labels per sample for each file
    """
    locations = Counter()
    rec_place = Counter()
    snrs = Counter()
    labels_count = Counter()
    for filename in wav_files:
        # Filename is e.g. .../CAFE-CAFE-1/sB/l060/n-05/CAFE-CAFE-1_sB_l060_n-05_i54300_x657e4.wav
        path = filename.split('/')
        snrs[path[-2]] += 1
        rec_place[path[-4]] += 1
        locations[path[-5]] += 1
    if labels_list is not None:
        for labels in labels_list:
            # go through each file
            unique, counts = np.unique(labels, return_counts=True)
            for possible_label in range(len(unique)):
                # loop through all unique labels, e.g. 0, 1, 2, 3, ...
                labels_count[unique[possible_label]] += counts[possible_label]
    print(snrs)
    print(labels_count)
    print(rec_place)
    print(locations)
    plt.bar(labels_count.keys(), labels_count.values())
    plt.show()
    fig = plt.figure()
    plt.bar(locations.keys(), locations.values())
    fig.autofmt_xdate(rotation=45)
    plt.show()
