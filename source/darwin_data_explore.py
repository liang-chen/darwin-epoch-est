
#! /usr/bin/python

import sys
import csv
import numpy as np
from math import log
from KLcluster import learn_from_KL
from myClass import mGaussian
from epoch_decode import estimate_epochs

def main():
    num_topics = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    metapath = "../data/metadata.csv"
    filepath = "../data/KLs/kl_dists{0}.csv".format(num_topics)
    filepath_topics = "../data/topics/topics{0}.csv".format(num_topics)
    
    meta_date = []
    with open(metapath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader, None)
        for row in reader:
            meta_date.append(row[2])
    
    names = []
    dat_rows = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader, None)
        for row in reader:
            names.append(row[0])
            dat_rows.append([float(x) for x in row[1:]])
    data_kl = np.array(dat_rows)

    names_new = []
    dat_rows_new = []
    with open(filepath_topics, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader, None)
        for row in reader:
            names_new.append(row[0])
            dat_rows_new.append([float(x) for x in row[1:]])

    data_topics = np.array(dat_rows_new)
    pairs, kGaussians = learn_from_KL(data_kl,data_topics,4)
    states = estimate_epochs(pairs, kGaussians, num_epochs)

    print '-'*20
    for i in xrange(len(states)):
        if i+1 < len(states):
            if states[i] != states[i+1]:
                print meta_date[i]

if __name__ == "__main__":
    main()
    
