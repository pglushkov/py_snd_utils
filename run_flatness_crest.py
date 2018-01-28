

import os
import sys
import numpy
import scipy.signal as sig
import scipy.io.wavfile as wav
import argparse


import utils.spectrum_proc_utils as utils_sp
import utils.sig_utils as utils_sig
import utils.tdomain_proc_utils as utils_td
import utils.plot_utils as utils_plot


def run_main():
    
    if len(sys.argv) <= 1:
        raise Exception("Need to specify input wav-file to process")
    
    wavname = sys.argv[1]
    
    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

    print("Will process file : {0}".format(wavname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate 
    signal = signal.reshape( (-1, 1) )


    crest_frame_size = 2048

    sflat_frame_size = 1024
    sflat_fft_size = int(2 ** numpy.ceil( numpy.log2(sflat_frame_size)))


    signal_for_crest = utils_sig.pad_to_multiple_of(signal, crest_frame_size, 0.0)
    crestchunks = utils_sig.cut_sig_into_chunks(signal_for_crest.T, crest_frame_size)
    crestfactor_vals = utils_td.get_crest_from_chunks(crestchunks)
 
    freq_grid, time_grid, sgram = sig.spectrogram(signal.squeeze(), fs=samplerate, 
        window = sig.get_window('boxcar', sflat_frame_size), nperseg = sflat_frame_size,
        noverlap = 0, nfft = sflat_fft_size, scaling = 'spectrum', mode = 'magnitude')
    
    sgram = sgram.T
 
    flatness = utils_sp.calc_spec_gram_flatness(sgram)

    crestfactor_vals.tofile('./tmp/crest_vals.bin')
    flatness.tofile('./tmp/flatness_vals.bin')


if __name__ == '__main__':
    run_main()
