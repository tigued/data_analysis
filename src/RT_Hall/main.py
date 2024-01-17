from ppms import main as ppms_main
from xrd import main as xrd_main
import sys
import argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("sample", type=str, help="sample name")
    argparse.add_argument("dirname", type=str, help="directory name")
    argparse.add_argument("ch_num", type=str, help="channel number")
    args = argparse.parse_args()
    sample = args.sample
    dirname = args.dirname
    ch_num = [int(i) for i in args.ch_num.split(",")]

    xrd_main(sample)
    ppms_main(sample, dirname, ch_num)
