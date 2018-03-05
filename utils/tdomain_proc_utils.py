

import numpy

import utils.sig_utils as utils_sig

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


def get_peak_to_peak(inp):
    minval = numpy.min(inp)
    maxval = numpy.max(inp)
    return maxval - minval


def get_peak_to_peak_from_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = get_peak_to_peak(chunks[k,:])
        res[k,0] = tmp
    return res


def get_energy(inp):
    return numpy.sum(inp ** 2)


def get_energy_from_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = get_energy(chunks[k,:])
        res[k,0] = tmp
    return res

def deriv(inp):
    assert(utils_sig.is_row(inp) or numpy.ndim(inp) == 1)
    if (numpy.ndim(inp) == 1):
        inp = inp.reshape( (1,-1) )

    inp_len = inp.shape[1]
    res = numpy.zeros( (1, inp_len - 2))

    for k in range(1, inp_len - 1):
        res[0, k-1] = (inp[0, k + 1] - inp[0, k - 1])/2.0

    return res

def get_deriv_from_chunks(chunks):
    num_chunks = chunks.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = deriv(chunks[k,:])
        res[k,0] = tmp
    return res
