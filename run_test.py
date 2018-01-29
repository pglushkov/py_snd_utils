
import os
import sys
import numpy
import scipy.signal as sig
import scipy.io.wavfile as wav


import utils.spectrum_proc_utils as utils_sp
import utils.sig_utils as utils_sig
import utils.tdomain_proc_utils as utils_td
import utils.plot_utils as utils_plot


def tst_sig_chunks():
    len = 13
    chunk_size = 8
    olap = 4

    sig = numpy.random.randn(1, len)

    print(sig.shape)
    print(sig)

    # case 0
    res = utils_sig.cut_sig_into_chunks(sig, chunk_size, overlap_size = olap, pad_zeros = False)
    print(res.shape)
    print(res)

    # case 1
    res = utils_sig.cut_sig_into_chunks(sig, chunk_size, overlap_size = olap)
    print(res.shape)
    print(res)

    # case 2
    res = utils_sig.cut_sig_into_chunks(sig, chunk_size)
    print(res.shape)
    print(res)

    # case 3
    res = utils_sig.cut_sig_into_chunks(sig, chunk_size, pad_zeros = False)
    print(res.shape)
    print(res)

def run_main():

    tst_sig_chunks()


if __name__ == '__main__':
    run_main()

