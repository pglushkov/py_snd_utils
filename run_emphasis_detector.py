
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
from run_spectral_change import estimate_sc_from_envelopes

def parse_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', required = True, type = str,
                       help = 'name of input wav-file (16k, 16bit PCM only!!!)')
    parse.add_argument('-o', required = True, type = str,
                       help = 'name of output bin-file, will contain values in float32 format')
    parse.add_argument('--cfg', required = False, type = str,
                       help = '(optional) name of JSON config-file with some expert setups for processing')
    parse.add_argument('-m', required = False, type = str,
                       help = '(optopmal) name of input mask-file against which we can evaluate detector')
    parse.add_argument('--mode', required = False, type = str,
                       help = '(optional) mode of operation \'file\' for processing 1 file, \'dir\' for '
                              'processing a directory. File is default' )
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
    res['f0_extr_thr'] = 0.1 # in std-devs
    res['f0_extr_len'] = 0.060 # in seconds
    res['f0_time_step'] = 0.005 # in seconds
    res['peak_to_peak_thr_std'] = 7.0
    res['out_dir'] = ''
    res['detect_hysteresis'] = 0.04 # seconds
    res['debug_mode'] = True # enable/disable additional logging and s#@t
    res['spec_change_threshold'] = 3.0
    res['spec_change_band_st'] = 500 # hz
    res['spec_change_band_end'] = 4500  # hz

    return res

def run_main_one_file(infile, outfile, maskfile, CFG):

    wavname = infile
    outname = os.path.join( CFG['out_dir'], outfile )

    if not os.path.exists(wavname):
        raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

    print("Will process file : {0}".format(wavname))
    print("Will write result to : {0}".format(outname))

    (detect, detect_time, dbg_stuff) = run_emp_detect(wavname, CFG, silent = True)

    (samplerate, signal) = wav.read(wavname)
    signal_time = numpy.arange(len(signal)) / samplerate

    plot_curves_y = [signal/(2.0 ** 15.0), detect]
    plot_curves_x = [signal_time, detect_time]
    plot_labels = ['input signal', 'detection mask']

    if maskfile is not None:
        (samplerate, mask) = wav.read(maskfile)
        mask = mask / (2.0 ** 15.0)
        mask_time = numpy.arange(len(mask)) / samplerate
        plot_curves_y.append(mask)
        plot_curves_x.append(mask_time)
        plot_labels.append('reference manual mask')

    detected_signal = signal * detect

    print('All done ...')
    print('Saving WAV result to {0}'.format(outname))
    wav.write(outname, samplerate, detected_signal)

    plotname = os.path.splitext(outname)[0] + '.png'
    utils_plot.plot_curves(plot_curves_y, plot_curves_x, labels = plot_labels, saveto = plotname)
    print('Saving plot to {0}'.format(plotname))

    if (CFG['debug_mode']):
        plot_debug_data(dbg_stuff, plot_curves_y, plot_curves_x, plot_labels, signal_time, outname,
                            CFG)

    print('All done, press any key ...')
    #input()

def run_main_proc_dir(indir, outdir, maskdir, CFG):

    assert (os.path.isdir(indir))
    if (maskdir is not None):
        assert (os.path.isdir(maskdir))
    if not os.path.isdir(outdir):
        print("WARNING: output directory {0} does not exist, creating it!".format(outdir))
        os.makedirs(outdir)

    wavfiles = os.listdir(indir)

    for fname in wavfiles:

        if os.path.splitext(fname)[1] != '.wav':
            continue

        wavname = os.path.join( indir, fname)
        outname = os.path.join( outdir, os.path.splitext(fname)[0] + '_RES.wav' )
        if maskdir is not None:
            maskfile = os.path.join(maskdir, os.path.splitext(fname)[0] + '_proc.wav')
        else:
            maskfile = None

        if not os.path.exists(wavname):
            raise Exception("Specified wavfile {0} does not seem to exist!".format(wavname))

        run_main_one_file(wavname, outname, maskfile, CFG)

def run_emp_detect(wavfile, config, silent = True):

    # ATTENTION!!! CURRENTLY NOT IMPLEMENTED!!!
    # need to work with
    # (DONE) 1) spectral change (basically envelope stability sort of)
    # (DONE) 2) peak-to-peak rate
    # (DONE) 3) syllable duration (basically a non-interrupted pitch segment)
    # (DONE) 4) pitch maxima (probably relatively to it's average value)

    (samplerate, signal) = wav.read(wavfile)
    sampleperiod = 1.0 / samplerate
    signal_time = numpy.arange(len(signal)) * sampleperiod
    signal = signal.reshape( (-1, 1) )
    signal = signal / (2.0 ** 15.0)
    signal_no_sil = utils_td.remove_silence(signal, 0.0001)
    std_no_sil = numpy.std(signal_no_sil)
    rms_no_sil = utils_td.get_rms(signal_no_sil)

    print(numpy.mean(signal_no_sil))
    print(std_no_sil)
    print(rms_no_sil)


    chunk_nsamples = int(config['chunk_size_samples'])
    olap_nsamples = int(config['overlap_samples'])

    fft_size = int(2 ** numpy.ceil( numpy.log2(chunk_nsamples)))
    env_size = int(fft_size / 2 + 1)

    sig_chunks = utils_sig.cut_sig_into_chunks(signal.T, chunk_nsamples, overlap_step = olap_nsamples,
        pad_zeros = True)
    sig_chunks_num = sig_chunks.shape[0]
    sig_chunks_tstep = olap_nsamples / samplerate
    sig_chunks_time = numpy.arange(sig_chunks_num) * sig_chunks_tstep

    wrld_res = run_world_by_reaper(wavfile, config['wrk_path'], config['reaper_path'], config['world_path'])

    sig_f0 = numpy.fromfile(wrld_res[0]).reshape( (-1, 1))
    sig_sp = numpy.fromfile(wrld_res[1]).reshape( (env_size, -1))
    sig_f0_time = numpy.arange(sig_f0.shape[0]) * config['f0_time_step']
    sig_f0_no_sil = utils_td.remove_silence(sig_f0, 0.0)
    f0_std_no_sil = numpy.std(sig_f0_no_sil)

    #print(sig_f0.shape)
    #print(sig_sp.shape)

    ### =================== PITCH EXTREMUMS
    f0_extr = utils_pitch.get_f0_extreme_areas(sig_f0, config['f0_extr_thr'],
                                               config['f0_extr_len'] / config['f0_time_step'])
    f0_low = f0_extr[0] * (f0_extr[0] > 0).astype('int')
    f0_high = f0_extr[1] * (f0_extr[1] > 0).astype('int')
    f0_extr = utils_td.perform_mvn_norm( (f0_low + f0_high), skip_zeros = True)
    # MY_DBG
    #utils_plot.plot_curves( [sig_f0 / numpy.max(sig_f0), f0_low, f0_high])

    ### =================== SPECTRAL CHANGE
    freq_step = samplerate / fft_size
    band_idx = [int(numpy.round(CFG['spec_change_band_st'] / freq_step)),
                int(numpy.round(CFG['spec_change_band_end'] / freq_step))]
    (sc_dfb, sc_time) = estimate_sc_from_envelopes(sig_sp.T, samplerate, 0.005 * samplerate, band = band_idx)
    sc_res = (numpy.abs(sc_dfb) < config['spec_change_threshold'] * std_no_sil).astype('float')

    sc_res = sc_dfb * sc_res
    # MY_DBG
    #utils_plot.plot_curves([signal, sc_res], [signal_time, sc_time])

    ### =================== PEAK-to_PEAK
    p2p = utils_td.get_peak_to_peak_from_chunks(sig_chunks)
    p2p_det = (p2p > config['peak_to_peak_thr_std'] * std_no_sil).astype('float') * p2p
    # MY_DBG
    #utils_plot.plot_curves( [signal, p2p], [signal_time, sig_chunks_time])

    ### =================== VOICED MASK
    voiced = (sig_f0.squeeze() > 0.0).astype('int')
    # MY_DBG
    #utils_plot.plot_curves( [sig_f0 / numpy.max(sig_f0), voiced], [sig_f0_time, sig_f0_time])

    ### =================== FINALIZING RESULTS
    DETECT_SC = numpy.interp(signal_time.squeeze(), sc_time.squeeze(), sc_res.squeeze())
    DETECT_VO = numpy.interp(signal_time.squeeze(), sig_f0_time.squeeze(), voiced.squeeze())
    DETECT_PP = numpy.interp(signal_time.squeeze(), sig_chunks_time.squeeze(), p2p_det.squeeze())
    DETECT_EX = numpy.interp(signal_time.squeeze(), sig_f0_time.squeeze(), f0_extr.squeeze())

    RESULT_MASK = (DETECT_SC > 0) * (DETECT_VO > 0) * (DETECT_PP > 0) * (DETECT_EX > 0)
    RESULT_MASK = update_detection_results(RESULT_MASK, samplerate, config['detect_hysteresis'])

    if not silent:
        utils_plot.plot_curves( [signal, RESULT_MASK], [signal_time, signal_time])

    SC_DFB = numpy.interp(signal_time.squeeze(), sc_time.squeeze(), sc_dfb.squeeze())
    P2P = numpy.interp(signal_time.squeeze(), sig_chunks_time.squeeze(), p2p.squeeze())

    dbg_stuff = {'spec_change_detect' : DETECT_SC, 'voiced_detect' : DETECT_VO, 'peak2peak_detect' : DETECT_PP,
        'f0-extreme_detect' : DETECT_EX, 'threshold_base' : std_no_sil, 'spec_change' : SC_DFB,
        'peak2peak' : P2P}

    return (RESULT_MASK, signal_time, dbg_stuff)

def update_detection_results(mask, samplerate, detect_hysteresis):
    assert(utils_sig.is_array(mask))
    hyst_step = int(numpy.round(detect_hysteresis * samplerate))
    sig_len = len(mask)
    idx = 0
    result = numpy.copy(mask)
    while(idx < sig_len):
        if (mask[idx] > 0.0):
            result[idx : idx + hyst_step] = 1.0
            idx += hyst_step
            continue
        idx += 1

    return result

def plot_debug_data(dbg_stuff, plot_curves_y, plot_curves_x, plot_labels, signal_time, outname, CFG):
    print('\n.................................')
    print('DEBUG mode enabled, will save some additional info: {0}'.format(str(dbg_stuff.keys())))
    base_plot_y = plot_curves_y
    del base_plot_y[1]
    base_plot_x = plot_curves_x
    del base_plot_x[1]
    base_plot_labels = plot_labels
    del base_plot_labels[1]

    threshold_base = dbg_stuff['threshold_base']

    plotname = os.path.splitext(outname)[0] + '_spec_change.png'
    thr_y = CFG['spec_change_threshold'] * threshold_base * numpy.ones( len(signal_time))
    utils_plot.plot_curves(base_plot_y + [dbg_stuff['spec_change'], thr_y],
                           base_plot_x + [signal_time, signal_time],
                           labels=base_plot_labels + ['spec_change', 'threshold'],
                           saveto=plotname)

    plotname = os.path.splitext(outname)[0] + '_voiced.png'
    utils_plot.plot_curves(base_plot_y + [dbg_stuff['voiced_detect']],
                           base_plot_x + [signal_time], labels=base_plot_labels + ['voiced'],
                           saveto=plotname)

    plotname = os.path.splitext(outname)[0] + '_p2p.png'
    thr_y = CFG['peak_to_peak_thr_std'] * threshold_base * numpy.ones(len(signal_time))
    utils_plot.plot_curves(base_plot_y + [dbg_stuff['peak2peak'], thr_y],
                           base_plot_x + [signal_time, signal_time],
                           labels=base_plot_labels + ['peak2peak', 'threshold'],
                           saveto=plotname)

    plotname = os.path.splitext(outname)[0] + '_f0_extr.png'
    utils_plot.plot_curves(base_plot_y + [dbg_stuff['f0-extreme_detect']],
                           base_plot_x + [signal_time], labels=base_plot_labels + ['f0-extreme'],
                           saveto=plotname)

if __name__ == '__main__':

    ARGS = parse_input_args()

    mode = ARGS.mode

    input = ARGS.i
    output = ARGS.o
    mask = ARGS.m

    if ARGS.cfg is None:
        CFG = get_default_config()
    else:
        CFG = get_config_from_json(ARGS.cfg)

    mode = ARGS.mode
    if mode is None:
        mode = 'file'

    if mode == 'file':
        # run everything to process one input file only
        run_main_one_file(input, output, mask, CFG)
    elif mode == 'dir':
        # run everything to process a set of files in specified dir
        run_main_proc_dir(input, output, mask, CFG)
    else:
        raise Exception('ERROR : some unknown mode scecified')
