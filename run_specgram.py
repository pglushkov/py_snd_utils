

import os
import sys
import numpy
import scipy.signal as sig
import scipy.io.wavfile as wav

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

    print("Will process file {0}".format(wavname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate 
    signal = signal.reshape( (-1, 1) )

    fft_size = 256
    nfilters = 15

    signal = utils_sig.pad_to_multiple_of(signal, fft_size, 0.0)
    sigchunks = utils_sig.cut_sig_into_chunks(signal.T, fft_size)
    spec_envs = utils_sp.get_spec_envelopes(sigchunks)
  
    # EXAMPLE: 
    #scipy.signal.spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
 
    freq_grid, time_grid, sgram = sig.spectrogram(signal.squeeze(), fs=samplerate, window = sig.get_window('boxcar', fft_size), nperseg = fft_size,
        noverlap = 0, nfft = fft_size, scaling = 'spectrum', mode = 'magnitude')
    
    sgram = sgram.T

    #print(sgram.shape)
    #print(sgram.dtype)
    #print(spec_envs.shape)
    #print(spec_envs.dtype)
        
    sgram.tofile('./tmp/py_sgram.bin')
    spec_envs.tofile('./tmp/my_sgram.bin')
    

if __name__ == '__main__':
    run_main()

