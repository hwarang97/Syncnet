import argparse
import os
import glob
from pathlib import Path
import pandas as pd
import subprocess
from multiprocessing import Pool
from shutil import rmtree
import time


# parser = argparse.ArgumentParser()
# parser.add_argument("--data-root", type=str, required=True)
# parser.add_argument("--logs", type=str, default="/home/shoh25/data_verification.csv")
# opt = parser.parse_args()


def main(args):
    i, opt = args
    print(opt)


if __name__ == "__main__":
    lst = [[1, 2, 3, 4, 5, 6],
           [3, 4, 56, 9]]
    # num_workers = os.cpu_count()
    num_workers = 4
    with Pool(num_workers) as pool:
        pool.map(main, enumerate(lst))
