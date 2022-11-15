import os
import timeit
#from utility import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM, GRU, TimeDistributed
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
import keras.backend as K

import numpy as np
import apuf_lib as ap
import challengeUtils as cts
import utils
import pdb
import random
import math
from InvertFunc import InvertFunctions
from apuf_lib import APUF, CRP


debug = False

training_set = CRP(np.zeros(1), np.zeros(1))

def prepareChallenges(length, nrof_chs, mode = 'SAC', bias = 0.9, ratio = 0.85, shuffle=True):
    print("Prepare for Challenges... Mode = {}".format(mode))
    if mode == 'RANDOM':
        return utils.genNChallenges(length, nrof_chs)
    elif mode == 'BIASED':
        chs = utils.genNChallenges(length, nrof_chs, b = bias)
        chs = np.unique(chs, axis=0)
        chs_len = len(chs)
        while chs_len < nrof_chs:
            v = nrof_chs - chs_len
            print("Re-generate {} data for redundant challenges...".format(v))
            addi_chs = utils.genNChallenges(length, v*2, b = bias)
            chs = np.concatenate([chs, addi_chs])
            chs = np.unique(chs, axis=0)
            chs_len = len(chs)
        chs = chs[:nrof_chs]
        return chs
    elif mode == 'B-ADD-RAND':
        nrof_bch = int(nrof_chs*ratio)
        nrof_rand_c = nrof_chs - nrof_bch
        rch = utils.genNChallenges(length, nrof_rand_c)
        bch = prepareChallenges(length, nrof_bch, mode = 'BIASED', bias = bias)
        barch = np.concatenate([bch, rch])
        
        if shuffle:
            barch = cts.chShuffle(barch)
        return barch
    elif mode == 'SAC':
        #rand.15
        return cts.prepareDiffChs2(length, nrof_chs, ratio=ratio)
    else:
        print("Exception in Function \"prepareChallenges\": No mode match.")
        pdb.set_trace()
        return cts.prepareDiffChs2(length, nrof_chs, ratio=ratio)

# 15,50 2 layers for CCA
def trainPUFModel(in_dim, data_size, nTriggers, X, Y, epoch, batch_size, train_test_ratio):
    model = Sequential()
    seqlen = int(int(math.pow(2, nTriggers)))
    stride = 1#seqlen//2
    
    X = X.reshape(-1, in_dim)
    Y = Y.flatten()

    data_size = data_size - seqlen
    test_size = data_size-train_size
    slid_X = np.array([X[i:i+seqlen] for i in np.arange(0, data_size, stride)])
    slid_Y = np.array([Y[i:i+seqlen] for i in np.arange(0, data_size, stride)])

    train_samples = int(len(slid_X)*train_test_ratio)
    slid_train_x = slid_X[:train_samples]
    slid_test_x = slid_X[train_samples:]
    slid_train_y = slid_Y[:train_samples]
    slid_test_y = slid_Y[train_samples:]
    pdb.set_trace()

    model.add(GRU(5, return_sequences=True, input_shape=(seqlen, in_dim)))
    model.add(GRU(15, return_sequences=True))
    model.add(TimeDistributed(Dense(15, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #model.add(Dense(1, activation='sigmoid'))
    model.summary()
        
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])
    results = model.fit(
     slid_train_x, slid_train_y,
     epochs= epoch,
     #verbose=1,
     batch_size = batch_size,
     #callbacks = [callback],
     shuffle=False,
     validation_data = (slid_test_x, slid_test_y)
    )

    scores = model.evaluate(test_x, test_y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model


# DNN attacks on APUF
np.random.seed(37487)
length = 128

nrof_chs = 1000000#2600000
nrof_tr_bits = 2
#xNetReduced = False#True
#dSWord = np.array([])#np.random.choice(2, length)

epoch = 20000
rand_ratio = 0.85
train_test_ratio = 0.9

# for T-Flip-Flop
seqlen = int(math.pow(2, nrof_tr_bits))
nrof_chs = (nrof_chs//seqlen)*seqlen
train_size = (int(nrof_chs*train_test_ratio)//seqlen)*seqlen
test_size = nrof_chs - train_size
batch_size = 2000

train_chs = prepareChallenges(length, train_size, mode = 'RANDOM', ratio=rand_ratio)
test_chs = prepareChallenges(length, test_size, mode = 'RANDOM', ratio=rand_ratio)

chs = np.concatenate([train_chs, test_chs])

x = ap.challengeTransform(chs, length, nrof_chs)
#apuf = APUF(length)
#res = apuf.getPufResponse(parity, nrof_chs, noisefree=True)

f = InvertFunctions(chs)
c_idx = np.random.choice(range(length), nrof_tr_bits, replace=False)
c_idx.sort()
y = f.TFF_AND_back(c_idx).flatten()

#training_set = CRP(train_x, y[:train_size])
#testing_set = CRP(test_x, y[test_size:])

#training_set = CRP(train_chs, y[:train_size])
#testing_set = CRP(test_chs, y[test_size:])

#pdb.set_trace()
model = trainPUFModel(length+1, nrof_chs, nrof_tr_bits, x, y, epoch, batch_size, train_test_ratio)

'''
# ==================================
# Heldout test
# ==================================
nrof_heldout = 20000

# for T-Flip-Flop
if invFunction == 'TFF_AND':
    nrof_heldout = (nrof_heldout//seqlen)*seqlen

h_chs = utils.genNChallenges(length, nrof_heldout)
h_pty = ap.challengeTransform(h_chs, length, nrof_heldout)
h_pty = h_pty.reshape(-1, seqlen, length+1)
#h_res = apuf.getPufResponse(h_pty, nrof_heldout, noisefree=True)

#h_noisy_res, h_ivt_idx, h_nIvt_idx = getPoisonedRes(h_chs, c_idx, h_res, fType = invFunction, getMasked=True, xReduced=xNetReduced, deviceWord = dSWord, seqlen = seqlen)
hf = InvertFunctions(train_chs)
h_y = hf.TFF_AND_back(c_idx, seqlen).flatten()


#pdb.set_trace()
pred = model.predict(h_pty)
y_pred = utils.softResToHard(pred).flatten()
heldout_acc = (y_pred==h_y).sum()/nrof_heldout

print("========Setting====== \n-Trigger bits={} \n-Number of CRPs={}\nRand_ratio={}".format(nrof_tr_bits, nrof_chs, rand_ratio))

print("[heldout_test] Prediction Acc = {}".format(heldout_acc))
pdb.set_trace()
'''


