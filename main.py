import numpy as np
from htk import readhtk
import matplotlib.pyplot as plt

P_05 = np.log(0.5)

PAU = ['pau']

NULA =  ['n','u','l','a']
JEDNA = ['j','e','d','n','a:']
DVA =   ['d','v','a']
DVJE =  ['d','v','j','e']
TRI =   ['P_','i:']
CTYRI = ['t_S','t','i','z','i:']
PJET =  ['p','j','e','t']
SEST =  ['S','e','s','t']
SEDM =  ['s','e','d','m']
OSM =   ['o','s','m']
DEVJET = ['d','e','v','j','e','t']

with open('dicos/phonemes') as f:
    PHONEMES = f.readlines()
    PHONEMES = list(map(lambda x: x[:-1], PHONEMES))


def plot_lhs(lhs):
    plt.pcolormesh(lhs.T)
    if lhs.shape[1] == 138:
        plt.yticks(np.arange(0,138,3), PHONEMES, fontsize=7)
    else:
        plt.yticks(np.arange(0,46,1), PHONEMES, fontsize=7)
    plt.show()


def get_hmm(lhs, ph):
    states = np.full((len(ph)), -np.inf, np.float64)
    states[0] = - P_05

    for i, col in enumerate(lhs):
        for j in np.arange(np.min([states.size-1, i]), 0, -1):
            p = col[PHONEMES.index(ph[j])]
            states[j] = np.max([states[j] + P_05 + p, states[j-1] + P_05 + p])
        states[0] = states[0] + P_05 + col[PHONEMES.index(ph[0])]

    return states


def get_hmm_2(lhs, ph):
    states = np.full((len(lhs), len(ph)), -np.inf, np.float64)
    states[0,0] = lhs[0,PHONEMES.index(ph[0])]

    for i, col in enumerate(lhs[1:], start=1):
        states[i,0] = states[i-1,0] + P_05 + col[PHONEMES.index(ph[0])]
        for j in range(1, len(ph)):
            p = col[PHONEMES.index(ph[j])]
            states[i, j] = np.max([states[i-1, j] + P_05 + p, states[i-1, j-1] + P_05 + p])

    return states


lhs = readhtk('eval/a30009b1.lik')
#lhs = lhs[0:210]

new_lhs = np.zeros((lhs.shape[0], len(PHONEMES)))
for i in range(46):
    new_lhs.T[i] = np.sum(lhs.T[i*3:(i*3)+2], axis=0)

#new_lhs = np.concatenate((new_lhs[:56], new_lhs[69:129], new_lhs[144:195]))
lhs = np.log(new_lhs)


CHUNK_SIZE = 120
STEP = 3
chunk_start = 0

ph_hmm_values = np.full((11), -np.inf, np.float64)
ph_hmm_values_2 = np.full((11), -np.inf, np.float64)

while chunk_start < len(lhs):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(lhs))
    print(str(chunk_start) + "\t" + str(chunk_end))
    if chunk_start >= len(lhs) - 32:
        break

    x = []
    y = []
    s = 0
    for i in np.arange(chunk_start+(2*STEP), chunk_end-STEP, STEP):
        ph_hmm_values[0] = get_hmm(lhs[chunk_start:i], NULA)[-1]
        ph_hmm_values[1] = get_hmm(lhs[chunk_start:i], JEDNA)[-1]
        ph_hmm_values[2] = get_hmm(lhs[chunk_start:i], DVA)[-1]
        ph_hmm_values[3] = get_hmm(lhs[chunk_start:i], TRI)[-1]
        ph_hmm_values[4] = get_hmm(lhs[chunk_start:i], CTYRI)[-1]
        ph_hmm_values[5] = get_hmm(lhs[chunk_start:i], PJET)[-1]
        ph_hmm_values[6] = get_hmm(lhs[chunk_start:i], SEST)[-1]
        ph_hmm_values[7] = get_hmm(lhs[chunk_start:i], SEDM)[-1]
        ph_hmm_values[8] = get_hmm(lhs[chunk_start:i], OSM)[-1]
        ph_hmm_values[9] = get_hmm(lhs[chunk_start:i], DEVJET)[-1]
        ph_hmm_values[10] = get_hmm(lhs[chunk_start:i], PAU)[-1]

        ph_hmm_values_2[0] = get_hmm(lhs[i:chunk_end], NULA)[-1]
        ph_hmm_values_2[1] = get_hmm(lhs[i:chunk_end], JEDNA)[-1]
        ph_hmm_values_2[2] = get_hmm(lhs[i:chunk_end], DVA)[-1]
        ph_hmm_values_2[3] = get_hmm(lhs[i:chunk_end], TRI)[-1]
        ph_hmm_values_2[4] = get_hmm(lhs[i:chunk_end], CTYRI)[-1]
        ph_hmm_values_2[5] = get_hmm(lhs[i:chunk_end], PJET)[-1]
        ph_hmm_values_2[6] = get_hmm(lhs[i:chunk_end], SEST)[-1]
        ph_hmm_values_2[7] = get_hmm(lhs[i:chunk_end], SEDM)[-1]
        ph_hmm_values_2[8] = get_hmm(lhs[i:chunk_end], OSM)[-1]
        ph_hmm_values_2[9] = get_hmm(lhs[i:chunk_end], DEVJET)[-1]
        ph_hmm_values_2[10] = get_hmm(lhs[i:chunk_end], PAU)[-1]

        v = np.max(ph_hmm_values) + np.max(ph_hmm_values_2)
        if i == chunk_start+STEP:
            s = v
        x.append(v)
        y.append(i)
    max_split = y[x.index(np.max(x))]
    if s == np.max(x):
        break

    chunk_start = max_split

plot_lhs(new_lhs)
#plt.plot(new_lhs[:,31])
#plt.show()
