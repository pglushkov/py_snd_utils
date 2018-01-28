
import numpy

def is_col(inp):
    return len(inp.shape) == 2 and inp.shape[1] == 1


def is_row(inp):
    return len(inp.shape) == 2 and inp.shape[0] == 1

def pad_to_multiple_of(signal, chunk_size, path_with = 0.0):
    num_frames = int(numpy.ceil( float(signal.shape[0]) / float(chunk_size) ))
    act_len = num_frames * chunk_size
    num_zeros = act_len - signal.shape[0]
    out = numpy.copy(signal)
    if (num_zeros > 0):
        out = numpy.vstack( (out, numpy.zeros( (num_zeros, 1) )) )

    return out

def cut_sig_into_chunks(signal, chunk_size):
    assert is_row(signal)
    assert (signal.shape[1] % chunk_size) == 0
    res = numpy.copy(signal).reshape( (-1, chunk_size) )
    return res

