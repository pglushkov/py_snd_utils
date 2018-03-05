
import numpy

def is_col(inp):
    return len(inp.shape) == 2 and inp.shape[1] == 1


def is_row(inp):
    return len(inp.shape) == 2 and inp.shape[0] == 1

def is_vector(inp):
    return len(inp.shape) == 2 and (inp.shape[0] == 1 or inp.shape[1] == 1)

def pad_to_multiple_of(signal, chunk_size, path_with = 0.0):
    num_frames = int(numpy.ceil( float(signal.shape[0]) / float(chunk_size) ))
    act_len = num_frames * chunk_size
    num_zeros = act_len - signal.shape[0]
    if (num_zeros > 0):
        out = numpy.vstack( (signal, numpy.zeros( (num_zeros, 1) )) )
    else:
        out = signal

    return out


def cut_sig_into_chunks(signal, chunk_size, overlap_step = 0, pad_zeros = True):
    assert is_row(signal)
    in_len = signal.shape[1]

    if pad_zeros:
        if overlap_step > 0:
            num_steps = int( numpy.ceil(float(in_len) / overlap_step) )
            total_size = num_steps * overlap_step + (chunk_size - overlap_step)
            num_zeros = total_size - in_len
            tmp = numpy.hstack( (signal, numpy.zeros((1, num_zeros))) )
            out = numpy.zeros( (num_steps, chunk_size))
            for k in range(num_steps):
                out[k,:] = tmp[0, k * overlap_step : (k * overlap_step) + chunk_size]
            return out
        else:
            num_steps = int( numpy.ceil(float(in_len) / chunk_size) )
            total_size = chunk_size * num_steps
            num_zeros = total_size - in_len
            out = numpy.hstack( (signal, numpy.zeros((1, num_zeros))) )
            assert (out.shape[1] % chunk_size) == 0
            return out.reshape((-1, chunk_size))
    else:
        if overlap_step > 0:
            num_steps = int( numpy.ceil(float(in_len - chunk_size) / overlap_step) )
            out = numpy.zeros( (num_steps, chunk_size))
            for k in range(num_steps):
                out[k,:] = signal[0, k * overlap_step : (k * overlap_step) + chunk_size]
            return out
        else:
            num_chunks = int(in_len // chunk_size)
            return signal[0, 0:num_chunks * chunk_size].reshape( (-1, chunk_size))

