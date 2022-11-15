import numpy as np
import apuf_lib as ap
import utils
import pdb
import random
from apuf_lib import APUF, CRP

debug = False

# bchal = np.random.choice(2, length*nrof_chs, p=[0.1, 0.9]).reshape(-1, length)
def prepareDiffChs(length, nrof_chs, ratio=0.5):
    nrof_bch = int(ratio*nrof_chs)
    #bchal = np.zeros(length).reshape(-1, length)
    bchal = np.random.choice(2, length).reshape(-1, length)
    while len(bchal) < nrof_bch:
        bchal = utils.genChallengesBySAC(length, len(bchal), chs=bchal)
        bchal = np.unique(bchal, axis=0)
        if (len(bchal)*length > 3000000) and (len(bchal) < nrof_bch):
            bchal = bchal[:50000]
            bchal = utils.genChallengesBySAC(length, len(bchal), chs=bchal)
            bchal = np.unique(bchal, axis=0)
            break
        print("bchal.shape = {}".format(bchal.shape))
    bchal = bchal[:nrof_bch]
    if debug:
        pdb.set_trace()
    if ratio != 1:
        nrof_rand_c = nrof_chs - nrof_bch
        rchal = utils.genNChallenges(length, nrof_rand_c)
        bchal = np.concatenate([bchal, rchal])
    #Shuffle
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    bchal = bchal[chidx]
    return bchal

def prepareDiffChs2(length, nrof_chs, ratio=0.5, shuffle = True):
    expand_limit = 3000
    nrof_bch = int(ratio*nrof_chs)
    #bchal = np.ones(length).reshape(-1, length)
    #bchal = np.zeros(length).reshape(-1, length)
    #bchal = np.random.choice(2, length, p=(0.9, 0.1)).reshape(-1, length)
    bchal = np.random.choice(2, length).reshape(-1, length)
    addchal = bchal.copy()
    print("initial_bchal = {}".format(bchal))
    while len(bchal) < nrof_bch:
        addchal = utils.genChallengesBySAC(length, len(addchal), chs=addchal)
        addchal = np.unique(addchal, axis=0)
        bchal = np.concatenate([bchal, addchal])
        bchal = np.unique(bchal, axis=0)
        if len(addchal) > expand_limit:
            sampledIdx = np.random.choice(len(bchal), expand_limit, replace=False)
            addchal = bchal[sampledIdx]
        print("bchal.shape = {}".format(bchal.shape))
    bchal = bchal[:nrof_bch]
    if debug:
        pdb.set_trace()
    if ratio != 1:
        nrof_rand_c = nrof_chs - nrof_bch
        rchal = utils.genNChallenges(length, nrof_rand_c)
        bchal = np.concatenate([bchal, rchal])

    #Shuffle
    if shuffle:
        bchal = chShuffle(bchal)
    return bchal

def chShuffle(chs):
    nrof_chs = len(chs)
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    return chs[chidx]

def HDDist(chs):
    nrof_chs = chs.shape[0]
    length = chs.shape[1]
    hd_hist = np.zeros(length+1)
    for i in range(nrof_chs):
        hd = np.logical_xor(chs[i], chs[i+1:]).sum(axis=1)
        maxHD = np.amax(hd)
        for j in range(maxHD):
            hd_hist[j] = np.count_nonzero(hd==j)
        if HDDist:
            print("HDDist. i={}".format(i))
    return hd_dist

def HDAvg(hddist):
    nrof_chs = hddist.sum()
    length = hddist.shape[0]
    hdweighted = hddist*np.arange(length+1)
    avgHD = hdweighted.sum()/nrof_chs
    return avgHD

'''
# not working
def prepareChallenges(length, nrof_chs, bias = 0.05):
    chs = utils.genNChallenges(length, nrof_chs//2, b=bias)
    chs_rand = utils.genNChallenges(length, nrof_chs//2)
    chs = np.concatenate([chs, chs_rand])
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    chs = chs[chidx]
    if debug:
        pdb.set_trace()
    return chs
'''
