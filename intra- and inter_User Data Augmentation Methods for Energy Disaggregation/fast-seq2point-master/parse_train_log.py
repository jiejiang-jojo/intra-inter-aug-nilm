""" A train log parser """
import csv
import sys
import re
from datetime import datetime


FIELDS = {
    'train': re.compile(r'train: (\d+\.\d+),'),
    'validate': re.compile(r'validate: (\d+\.\d+)$'),
    
}

def extract(ptn, line):
    """ Extract the pattern in the line """
    match = ptn.search(line)
    return match.group(1)


def parse_log(filename):
    """ Parse training log file to output csv """
    with open(filename) as fin:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS.keys())
        writer.writeheader()
        for line in fin:
            try:
                if 'train' in line:
                    row = {}
                    row['train'] = extract(FIELDS['train'], line)
                    
                elif 'validate' in line:
                    row['validate'] = extract(FIELDS['validate'], line)
                elif '------------' in line:
                    writer.writerow(row)
            except Exception:
                pass

def console():
    """ Run from commandline """
    if len(sys.argv) < 2:
        print('Usage: parse_train_log.py <logfile>')
        sys.exit(-1)
    parse_log(sys.argv[1])

if __name__ == "__main__":
    console()
