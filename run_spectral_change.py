
import os
import sys
import numpy
import scipy.signal as sig
import scipy.io.wavfile as wav

import python_speech_features as psf

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
    fbank_envs = utils_sp.get_mel_fb_curves(spec_envs, samplerate, nfilters)

    timestep = float(fft_size) / float(samplerate)
    (fbank_envs_py, _) = psf.fbank(signal ,samplerate=samplerate ,winlen=timestep ,winstep=timestep,
                                   nfilt=nfilters ,nfft=fft_size ,lowfreq=0 ,highfreq=None ,preemph=0)

    # simple_plot(signal, numpy.arange(signal.shape[0]) * sampleperiod)
    # simple_plot(fbank_envs[30,:])
    # simple_plot(fbank_envs_py[30,:])

    print(fbank_envs.shape)
    print(fbank_envs_py.shape)

    print(fbank_envs.dtype)
    print(fbank_envs_py.dtype)

    fbank_envs.tofile('./tmp/my_fbank.bin')
    fbank_envs_py.tofile('./tmp/py_fbank.bin')


if __name__ == '__main__':
    run_main()

