

import numpy


def get_rms(inp):
    rms = numpy.sqrt( numpy.sum(inp ** 2) / len(inp))
    return rms

def get_crest_factor(inp, rms = None):
    if rms is None:
        rms = get_rms(inp)
    peak = numpy.max(numpy.abs(inp))
    return peak / rms

def get_crest_from_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    rms = get_rms(chunks.flatten())
    for k in range(num_chunks):
        tmp = get_crest_factor(chunks[k,:], rms = rms)
        res[k,0] = tmp
    return res

