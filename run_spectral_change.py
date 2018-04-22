
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
import utils.misc_utils as utils_misc
import utils.reaper_utils as utils_reaper

def run_main_sgram_env():

    if len(sys.argv) <= 1:
        raise Exception("Need to specify input wav-file to process")

    wavname = sys.argv[1]

    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

    print("Will process file {0}".format(wavname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate
    signal = signal.reshape( (-1, 1) )

    fft_size = 64
    nfilters = 15

    signal = utils_sig.pad_to_multiple_of(signal, fft_size, 0.0)
    sigchunks = utils_sig.cut_sig_into_chunks(signal.T, fft_size)
    spec_envs = utils_sp.get_spec_envelopes(sigchunks)
    fbank_envs = utils_sp.get_mel_fb_curves(spec_envs, samplerate, nfilters)

    timestep = float(fft_size) / float(samplerate)
    (fbank_envs_py, _) = psf.fbank(signal ,samplerate=samplerate ,winlen=timestep ,winstep=timestep,
                                   nfilt=nfilters ,nfft=fft_size ,lowfreq=0 ,highfreq=None ,preemph=0)

    SIG_DUR = sampleperiod * signal.shape[0]
    SIG_X = numpy.arange(0, SIG_DUR, sampleperiod)

    dfb, D_FB_X, _ = estimate_sc_from_envelopes(fbank_envs, samplerate, fft_size)

    #utils_plot.simple_plot(signal, SIG_X)
    #utils_plot.plot_curves( [signal, fbank_envs[:,1]], [SIG_X, FB_X] )
    #utils_plot.plot_curves([signal, deriv], [SIG_X, D_FB_X])
    utils_plot.plot_curves([signal, dfb], [SIG_X, D_FB_X])

def run_main_world_env(fft_size, tstep):

    if len(sys.argv) < 5:
        raise Exception("Need to specify input wav, sp, mask, f0 files to process")

    wavname = sys.argv[1]
    spname = sys.argv[2]
    maskname = sys.argv[3]
    f0name = sys.argv[4]

    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))
    if not os.path.exists(spname):
        raise Exception("Specified sp-file {0} does not seem to exist!".format(spname))
    if not os.path.exists(maskname):
        raise Exception("Specified mask {0} does not seem to exist!".format(maskname))
    if not os.path.exists(f0name):
        raise Exception("Specified mask {0} does not seem to exist!".format(f0name))

    print("Will process file {0}".format(wavname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate
    signal = signal.reshape( (-1, 1) ) / (2.0 ** 15.0)

    (_, mask) = wav.read(maskname)
    mask = mask.reshape( (-1, 1) ) / (2.0 ** 15.0)

    SIG_DUR = sampleperiod * signal.shape[0]
    SIG_X = numpy.arange(0, SIG_DUR, sampleperiod)

    MASK_DUR = sampleperiod * mask.shape[0]
    MASK_X = numpy.arange(0, MASK_DUR, sampleperiod)

    f0data = numpy.fromfile(f0name, dtype = 'float64')
    f0data /= numpy.max(numpy.abs(f0data))
    F0_DUR = tstep * len(f0data)
    F0_X = numpy.arange(0, F0_DUR, tstep)

    spvals = numpy.fromfile(spname, dtype = 'float64')
    fbank_envs = spvals.reshape( (-1, int(fft_size/2 + 1)) )

    dfb, D_FB_X, _ = estimate_sc_from_envelopes(fbank_envs, samplerate, int(tstep * samplerate))
    dfb /= numpy.max(numpy.abs(dfb))

    #utils_plot.simple_plot(fbank_envs[100,:])
    #utils_plot.plot_curves([signal, mask, dfb], [SIG_X, MASK_X, D_FB_X])
    utils_plot.plot_curves([f0data, mask, dfb], [F0_X, MASK_X, D_FB_X])

def run_main_reaper_pm_env(fft_time_step, tstep):

    if len(sys.argv) < 5:
        raise Exception("Need to specify input wav, pm, mask, f0 files to process")

    wavname = sys.argv[1]
    pmname = sys.argv[2]
    maskname = sys.argv[3]
    pmtxtname = sys.argv[4]

    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))
    if not os.path.exists(pmname):
        raise Exception("Specified pitch-marks file {0} does not seem to exist!".format(pmname))
    if not os.path.exists(maskname):
        raise Exception("Specified mask {0} does not seem to exist!".format(maskname))
    if not os.path.exists(pmtxtname):
        raise Exception("Specified pitch-marks ACII file {0} does not seem to exist!".format(pmtxtname))

    print("Will process file {0}".format(wavname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate
    signal = signal.reshape( (-1, 1) ) / (2.0 ** 15.0)

    (_, mask) = wav.read(maskname)
    mask = mask.reshape( (-1, 1) ) / (2.0 ** 15.0)

    fft_size = utils_misc.nextpow2(samplerate * fft_time_step)

    SIG_DUR = sampleperiod * signal.shape[0]
    SIG_X = numpy.arange(0, SIG_DUR, sampleperiod)

    MASK_DUR = sampleperiod * mask.shape[0]
    MASK_X = numpy.arange(0, MASK_DUR, sampleperiod)

    pmvals = numpy.fromfile(pmname, dtype = 'float64')
    pm_chunks = pmvals.reshape( (-1, fft_size) )
    pm_envs = utils_sp.get_spec_envelopes(pm_chunks)

    dfb, D_FB_X, _ = estimate_sc_from_envelopes(pm_envs, samplerate, fft_size)
    dfb /= numpy.max(numpy.abs(dfb))

    (pmarks, _) = utils_reaper.read_pm_from_file(pmtxtname)
    D_FB_X = pmarks[1:-1, 0]

    # MY_DBG
    #print(dfb.shape)

    #utils_plot.plot_curves([ pm_chunks[100,:], pm_chunks[101,:], pm_chunks[102,:] ])
    utils_plot.plot_curves([signal, mask, dfb], [SIG_X, MASK_X, D_FB_X])
    #print(pm_envs.shape)
    #print(dfb)
    #utils_plot.simple_plot(dfb.squeeze(), D_FB_X)
    #utils_plot.plot_curves([f0data, mask, dfb], [F0_X, MASK_X, D_FB_X])

def estimate_sc_from_envelopes(fbank_envs, samplerate, tstep_samples, band = None):

    # expect 'fbank_envs' to be [num_chunks, num_bins] shape

    sampleperiod = 1.0 / samplerate
    FB_STEP = sampleperiod * tstep_samples
    FB_DUR = fbank_envs.shape[0] * FB_STEP
    FB_X = numpy.arange(0, FB_DUR, FB_STEP)

    # 1st deriv
    if band is None:
        dfb = numpy.zeros(fbank_envs.shape[0] - 2).reshape( (1,-1) )
        for k in range(fbank_envs.shape[1]):
            dfb += numpy.abs(utils_td.deriv(fbank_envs[:,k].T))
    else:
        assert(len(band) == 2)
        dfb = numpy.zeros(fbank_envs.shape[0] - 2).reshape( (1,-1) )
        # MY_DBG
        #print(dfb.shape)
        for k in range(band[0], band[1]):
            dfb += numpy.abs(utils_td.deriv(fbank_envs[:, k].T))
    #dfb /= fbank_envs.shape[1]
    D_FB_X = numpy.arange(FB_STEP, FB_DUR - FB_STEP, FB_STEP)
    if len(D_FB_X) > dfb.shape[1]:
        D_FB_X = D_FB_X[0:dfb.shape[1]]

    D_FB_X = D_FB_X.reshape( dfb.shape )

    # # 2nd deriv
    # dfb = numpy.zeros(fbank_envs.shape[0] - 4).reshape( (1,-1) )
    # for k in range(fbank_envs.shape[1]):
    #     tmp = utils_td.deriv(fbank_envs[:, k].T)
    #     dfb += utils_td.deriv(tmp)
    # dfb /= fbank_envs.shape[1]
    # D_FB_X = numpy.arange(FB_STEP*2, FB_DUR - FB_STEP*2, FB_STEP)

    #dfb = numpy.diff(fbank_envs[:,2])
    #D_FB_X = numpy.arange( 0, FB_DUR - FB_STEP, FB_STEP )

    #return (deriv, D_FB_X)
    return (dfb, D_FB_X)

if __name__ == '__main__':

    #run_main_sgram_env()

    #run_main_world_env(1024, 0.005)

    fft_time_step = 0.01
    run_main_reaper_pm_env(fft_time_step, 0.005)
