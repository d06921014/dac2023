import numpy as np
import pandas as pd
import apuf_lib as ap
import pdb
import random
import utils
from apuf_lib import APUF, XORAPUF, CRP
import matplotlib.pyplot as plt
from InvertFunc import InvertFunctions

debug = False
save = False



def getAAPUFResponse(apuf, chs, c_idx, tsType='XOR'):
    nrof_chs = chs.shape[0]
    length = chs.shape[1]
    parity = ap.challengeTransform(chs, length, nrof_chs)
    res = apuf.getPufResponse(parity, nrof_chs, noisefree=True)
    f = InvertFunctions(chs)
    mask = np.zeros(nrof_chs)
    if tsType == 'XOR':
        mask = f.XOR(c_idx)
    elif tsType == 'ANDOR':
        mask = f.ANDOR(c_idx.reshape(-1,2))
    elif tsType == 'TFF_AND':
        mask = f.TFF_AND(c_idx).flatten()
    elif tsType == 'AND':
        mask = f.AND(c_idx)
    elif tsType == 'NONE':
        pass
    else:
        print('getAAPUFResponse: No Trigger Type matched. set Trigger Type to \'XOR\'')
        mask = f.XOR(c_idx)
    
    ivt_idx = np.where(mask==1)[0]
    print("Inversion Function:{}\nc_idx = {}\nInverted_idx = {}, shape={}".format(tsType, c_idx, ivt_idx, ivt_idx.shape))
    poised_res = getPoisonedRes(res, ivt_idx)
    return poised_res

def getPoisonedRes(r, invert_idx):
    p_res = r.copy()
    for i in invert_idx:
       p_res[i] = not p_res[i]
    return p_res

def showPlt(length, trans, nchal):
    plt.ylim(0, 1)
    plt.plot(np.arange(length), trans/nchal)
    plt.show()

if __name__ == "__main__":

    length = 128
    nchal = 1280
    nrof_tr_bits = 8
    trigger_type = 'XOR'#'NONE'#'XOR'

    chs = utils.genNChallenges(length, nchal)
    chflip = np.tile(chs, (length, 1, 1))

    # for each index in filp array, for each challenge, flip the index-th bit
    for lidx, cs in enumerate(chflip):
        for nidx, c in enumerate(cs):
            chflip[lidx, nidx, lidx] = not c[lidx]
    #pdb.set_trace()
    sacfpRes = np.zeros((length, nchal))
    trans = np.zeros((length))
    #======================
    # init APUF   
    np.random.seed(11)
    #apuf = APUF(length)
    apuf = XORAPUF(length, 4)
    #ts_idx = np.array(random.sample(range(length), nrof_tr_bits))
    ts_idx = np.array([2,22,31,32,72,90,100,115])
    fpres = getAAPUFResponse(apuf, chs, ts_idx, tsType=trigger_type)
    #=====================
    for i in range(length):
        print("{}-th index...".format(i))
        #ap.dfpSelectedByChallengeSft(apuf, chflip[i], nchal, length)
        sacfpRes = getAAPUFResponse(apuf, chflip[i], ts_idx, tsType=trigger_type)
        #sacfpRes[i] = ap.dualFlip(apuf, chflip[i], nchal, length, np.arange(56,64))
        #pdb.set_trace()
        trans[i] = (sacfpRes != fpres).sum()
    showPlt(length, trans, nchal)
    pdb.set_trace()
    np.savetxt("output_transition/{}-{}-transprob.csv".format(nrof_tr_bits, trigger_type), trans.reshape(length,1)/nchal, fmt="%.4f", delimiter=',')
