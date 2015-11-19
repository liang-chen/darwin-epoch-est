
#! /usr/bin/python

import sys
import csv
import numpy as np


def main():
    filepath = sys.argv[1]

    names = []
    dat_rows = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader, None)
        for row in reader:
            names.append(row[0])
            dat_rows.append(row[1:])
        
    data = np.array(dat_rows)
    print data

if __name__ == "__main__":
    main()
    
