
import os
import sys
import numpy
import scipy.signal as sig
import scipy.io.wavfile as wav
import argparse
import json


import utils.spectrum_proc_utils as utils_sp
import utils.sig_utils as utils_sig
import utils.tdomain_proc_utils as utils_td
import utils.plot_utils as utils_plot
import utils.pitch_utils as utils_pitch

from run_world_by_reaper import run_world_by_reaper

def parse_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', required = True, type = str,
                       help = 'name of input wav-file (16k, 16bit PCM only!!!)')
    parse.add_argument('-o', required = True, type = str,
                       help = 'name of output bin-file, will contain values in float32 format')
    parse.add_argument('--cfg', required = False, type = str,
                       help = '(optional) name of JSON config-file with some expert setups for processing')
    args = parse.parse_args()
    return args

def get_config_from_json(filename):
    with open(filename) as cfg_file:
        jsoncfg = json.load(cfg_file)

    return jsoncfg

def get_default_config():
    res = {}
    res['chunk_size_samples'] = 1024
    res['overlap_samples'] = 80
    res['wrk_path'] = './tmp/'
    res['reaper_path'] = './bin_utils/reaper'
    res['world_path'] = './bin_utils/world_analysis'

    return res

def run_main():

    # ATTENTION!!! CURRENTLY NOT IMPLEMENTED!!!
    # need to work with
    # 1) spectral change (basically envelope stability sort of)
    # 2) peak-to-peak rate
    # 3) syllable duration (basically a non-interrupted pitch segment)
    # 4) pitch maxima (probably relatively to it's average value)

    ARGS = parse_input_args()

    # DBG
    # print(ARGS)

    wavname = ARGS.i
    outname = ARGS.o

    if ARGS.cfg is None:
        CFG = get_default_config()
    else:
        CFG = get_config_from_json(ARGS.cfg)

    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

    print("Will process file : {0}".format(wavname))
    print("Will write result to : {0}".format(outname))

    (samplerate, signal) = wav.read(wavname)
    sampleperiod = 1.0 / samplerate
    signal = signal.reshape( (-1, 1) )


    chunk_nsamples = int(CFG['chunk_size_samples'])
    olap_nsamples = int(CFG['overlap_samples'])

    fft_size = int(2 ** numpy.ceil( numpy.log2(chunk_nsamples)))
    env_size = int(fft_size / 2 + 1)


    sig_chunks = utils_sig.cut_sig_into_chunks(signal.T, chunk_nsamples, overlap_step = olap_nsamples,
        pad_zeros = True)

    wrld_res = run_world_by_reaper(wavname, CFG['wrk_path'], CFG['reaper_path'], CFG['world_path'])

    sig_f0 = numpy.fromfile(wrld_res[0]).reshape( (-1, 1))
    sig_sp = numpy.fromfile(wrld_res[1]).reshape( (env_size, -1))

    f0_extr = utils_pitch.get_f0_extreme_areas(sig_f0)

    utils_plot.plot_curves( [sig_f0, f0_extr[0], f0_extr[1]])

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
    # simple_plot(signal, SIG_X)
    # simple_plot(CREST_Y, SIG_X)
    # simple_plot(SFLAT_Y, SIG_X)

    RESULT_MASK = (CREST_Y > ARGS.crest_thr) * (SFLAT_Y >= ARGS.sflat_thr_down) * (SFLAT_Y <= ARGS.sflat_thr_up)
    # DBG
    utils_plot.simple_plot(RESULT_MASK, SIG_X)

    RESULT_MASK = RESULT_MASK.astype('float32')
    RESULT_MASK.tofile(outname)


if __name__ == '__main__':
    run_main()
