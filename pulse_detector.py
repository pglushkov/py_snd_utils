

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

def parse_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', required = True, type = str, 
            help = 'name of input wav-file (16k, 16bit PCM only!!!)')
    parse.add_argument('-o', required = True, type = str, 
            help = 'name of output bin-file, will contain values in float32 format')
    parse.add_argument('--crest_size', required = False, type = int, default = 20,
            help = 'integer, size of window in msec for crest-factor estimation, default 50') 
    parse.add_argument('--sflat_size', required = False, type = int, default = 25,
            help = 'integer, size of window in msec for spectral flatness estimation, default 25')
    parse.add_argument('--crest_thr', required = False, type = float, default = 2.0,
            help = 'float, threshold for crest-factor based peak detection')
    parse.add_argument('--sflat_thr_down', required = False, type = float, default = 0.35,
            help = 'float, lower boundary for spectral flatness based detector')
    parse.add_argument('--sflat_thr_up', required = False, type = float, default = 0.7,
            help = 'float, upper boundary for spectral flatness based detector')
    parse.add_argument('--hyst', required = False, type = int, default = 3,
            help = 'integer, hysteresis in frames - detection is valid only if spectral-flatness detector detects this number of consequent frames')
    args = parse.parse_args()
    return args 

def run_main():
    
    
    ARGS = parse_input_args()

    # DBG
    #print(ARGS)

    wavname = ARGS.i
    outname = ARGS.o
    
    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

    print("Will process file : {0}".format(wavname))
    print("Will write result to : {0}".format(outname)) 

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate 
    signal = signal.reshape( (-1, 1) )


    crest_frame_size = int(samplerate * float(ARGS.crest_size) / 1000.0)

    sflat_frame_size = int(samplerate * float(ARGS.sflat_size) / 1000.0)
    sflat_fft_size = int(2 ** numpy.ceil( numpy.log2(sflat_frame_size)))


    signal_for_crest = utils_sig.pad_to_multiple_of(signal, crest_frame_size, 0.0)
    crestchunks = utils_sig.cut_sig_into_chunks(signal_for_crest.T, crest_frame_size)
    crestfactor_vals = utils_td.get_crest_from_chunks(crestchunks)
 
    freq_grid, time_grid, sgram = sig.spectrogram(signal.squeeze(), fs=samplerate, 
        window = sig.get_window('boxcar', sflat_frame_size), nperseg = sflat_frame_size,
        noverlap = 0, nfft = sflat_fft_size, scaling = 'spectrum', mode = 'magnitude')
    
    sgram = sgram.T
 
    flatness = utils_sp.calc_spec_gram_flatness(sgram)

    crestperiod = crest_frame_size * sampleperiod
    sflatperiod = sflat_frame_size * sampleperiod
    SIG_X = numpy.arange(0, sampleperiod * len(signal), sampleperiod)
    CREST_X = numpy.arange(0, crestperiod * len(crestfactor_vals), crestperiod)
    SFLAT_X = numpy.arange(0, sflatperiod * len(flatness), sflatperiod)


    CREST_Y = numpy.interp(SIG_X, CREST_X, crestfactor_vals.squeeze())
    SFLAT_Y = numpy.interp(SIG_X, SFLAT_X, flatness.squeeze())    

    # DBG
    #simple_plot(signal, SIG_X)
    #simple_plot(CREST_Y, SIG_X)
    #simple_plot(SFLAT_Y, SIG_X)

    RESULT_MASK = (CREST_Y > ARGS.crest_thr) * (SFLAT_Y >= ARGS.sflat_thr_down) * (SFLAT_Y <= ARGS.sflat_thr_up)
    #DBG
    utils_plot.simple_plot(RESULT_MASK, SIG_X)

    RESULT_MASK = RESULT_MASK.astype('float32')
    RESULT_MASK.tofile(outname)


if __name__ == '__main__':
    run_main()
