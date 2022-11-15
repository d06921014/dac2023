import numpy as np
import apuf_lib as ap
import argparse
import pdb

debug = False

length = 128
nchal = 100
pat_v = np.zeros(length) 

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--length", help="Length of a PUF", type=int, default=length)
parser.add_argument("-n", "--noch", help="Number of challenges", type=int, default=nchal)
parser.add_argument("-sp", "--pattern", help="Pattern vector with HW(1)", type=int, default=0)
args = parser.parse_args()

pat_v[args.pattern]=1

chs = utils.genNChallenges(length = args.length, nrows=args.noch)

print(pat_v)
