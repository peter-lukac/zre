"""
HMM decoder for number phoneme likelyhood array decoding
author: Peter Lukac
login: xlukac11
April 2020
"""

import numpy as np
from htk import readhtk
import matplotlib.pyplot as plt
import sys

P_05 = np.log(0.5)

PAU = ['pau']
NULA =  ['n','u','l','a']
JEDNA = ['j','e','d','n','a:']
DVA =   ['d','v','a']
DVJE =  ['d','v','j','e']
TRI =   ['t','P_','i']
CTYRI = ['t_S','t','i','z','i']
PJET =  ['p','j','e','t']
SEST =  ['S','e','s','t']
SEDM =  ['s','e','d','m']
OSM =   ['o','s','m']
OSUM =  ['o','s','u','m']
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


def get_max_hmm(lhs, start, mid, stop, only_start=False):
    start_values = np.full((11), -np.inf, np.float64)
    end_values = np.full((11), -np.inf, np.float64)

    start_values[0] = get_hmm(lhs[start:mid], NULA)[-1]
    start_values[1] = get_hmm(lhs[start:mid], JEDNA)[-1]
    start_values[2] = np.max([get_hmm(lhs[start:mid], DVA)[-1], get_hmm(lhs[start:mid], DVJE)[-1]])
    start_values[3] = get_hmm(lhs[start:mid], TRI)[-1]
    start_values[4] = get_hmm(lhs[start:mid], CTYRI)[-1]
    start_values[5] = get_hmm(lhs[start:mid], PJET)[-1]
    start_values[6] = get_hmm(lhs[start:mid], SEST)[-1]
    start_values[7] = get_hmm(lhs[start:mid], SEDM)[-1]
    start_values[8] = np.max([get_hmm(lhs[start:mid], OSM)[-1], get_hmm(lhs[start:mid], OSUM)[-1]])
    start_values[9] = get_hmm(lhs[start:mid], DEVJET)[-1]
    start_values[10] = get_hmm(lhs[start:mid], PAU)[-1]

    if only_start is True:
        return start_values
    
    end_values[0] = get_hmm(lhs[mid:stop], NULA)[-1]
    end_values[1] = get_hmm(lhs[mid:stop], JEDNA)[-1]
    end_values[2] = np.max([get_hmm(lhs[mid:stop], DVA)[-1], get_hmm(lhs[mid:stop], DVJE)[-1]])
    end_values[3] = get_hmm(lhs[mid:stop], TRI)[-1]
    end_values[4] = get_hmm(lhs[mid:stop], CTYRI)[-1]
    end_values[5] = get_hmm(lhs[mid:stop], PJET)[-1]
    end_values[6] = get_hmm(lhs[mid:stop], SEST)[-1]
    end_values[7] = get_hmm(lhs[mid:stop], SEDM)[-1]
    end_values[8] = np.max([get_hmm(lhs[mid:stop], OSM)[-1], get_hmm(lhs[mid:stop], OSUM)[-1]])
    end_values[9] = get_hmm(lhs[mid:stop], DEVJET)[-1]
    end_values[10] = get_hmm(lhs[mid:stop], PAU)[-1]

    return start_values, end_values


if len(sys.argv) != 2:
    print("Usage: main.py FILE")
    exit(1)

lhs = readhtk(sys.argv[1])

new_lhs = np.zeros((lhs.shape[0], len(PHONEMES)))
for i in range(46):
    new_lhs.T[i] = np.sum(lhs.T[i*3:(i*3)+2], axis=0)

lhs = np.log(new_lhs)


CHUNK_SIZE = 120
STEP = 3
chunk_start = 0


while chunk_start < len(lhs):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(lhs))
    if chunk_start >= len(lhs) - 20:
        break

    max_value = []
    max_index = []
    max_label = []
    s = 0
    for i in np.arange(chunk_start+(2*STEP), chunk_end-STEP, STEP):
        ph_hmm_values, ph_hmm_values_2 = get_max_hmm(lhs, chunk_start, i, chunk_end)

        v = np.max(ph_hmm_values) + np.max(ph_hmm_values_2)
        max_label.append(np.argmax(ph_hmm_values))
        if i == chunk_start+(2*STEP):
            s = v
        max_value.append(v)
        max_index.append(i)

    max_split = max_index[max_value.index(np.max(max_value))]
    if max_label[max_value.index(np.max(max_value))] == 10:
        if max_split - chunk_start >= 70:
            print(np.argmax(get_max_hmm(lhs, chunk_start, max_split, 0, True)[0:10]), end = '')
            print(" ", end = '')
    else:
        print(max_label[max_value.index(np.max(max_value))], end = '')
        print(" ", end = '')
    if s == np.max(max_value):
        chunk_start += (2*STEP)
    chunk_start = max_split

print()