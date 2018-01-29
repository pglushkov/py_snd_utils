

import numpy


def get_rms(inp):
    rms = numpy.sqrt( numpy.sum(inp ** 2) / len(inp))
    return rms


def get_crest_factor(inp, rms = None):
    if rms is None:
        rms = get_rms(inp)
    peak = numpy.max(numpy.abs(inp))
    return peak / rms


def get_crest_from_chunks(chunks, use_global_rms = True):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )

    if (use_global_rms):
        rms = get_rms(chunks.flatten())
    else:
        rms = None

    for k in range(num_chunks):
        tmp = get_crest_factor(chunks[k,:], rms = rms)
        res[k,0] = tmp
    return res


def get_peak_to_peak_ratio(inp):
    minval = numpy.min(inp)
    maxval = numpy.max(inp)
    return maxval - minval


def get_peak_to_peak_from_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = get_peak_to_peak_ratio(chunks[k,:])
        res[k,0] = tmp
    return res


def get_energy(inp):
    return numpy.sum(inp ** 2)


def get_energy_for_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = get_energy(chunks[k,:])
        res[k,0] = tmp
    return res


