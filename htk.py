import numpy as np
import struct, sys, re, os
import warnings
import math

WAVEFORM = 0
IREFC = 5
DISCRETE = 10

_C = 0x0002000
_K = 0x0010000

parms16bit = [WAVEFORM, IREFC, DISCRETE]

def readhtk(file, return_parmKind_and_sampPeriod=False):
    try:
        fh = open(file,'rb')
    except TypeError:
        fh = file
    try:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
        m = np.frombuffer(fh.read(nSamples*sampSize), 'i1')
        pk = parmKind & 0x3f
        if pk in parms16bit:
            m = m.view('>h').reshape(nSamples,sampSize/2)
        elif parmKind & _C:
            scale, bias = m[:sampSize*4].view('>f').reshape(2,sampSize/2)
            m = (m.view('>h').reshape(nSamples,int(sampSize/2))[4:] + bias) / scale
        else:
            m = m.view('>f').reshape(nSamples,int(sampSize/4))
        if pk == IREFC:
            m = m / 32767.0
        if pk == WAVEFORM:
            m = m.ravel()
        if parmKind & _K:
            fh.read(1)
    finally:
        if fh is not file: fh.close()
    return m if not return_parmKind_and_sampPeriod else (m, parmKind, sampPeriod/1e7)
