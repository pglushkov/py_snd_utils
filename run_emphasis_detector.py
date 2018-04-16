
import os
import sys
import numpy
import scipy.signal as sci_sig
import scipy.io.wavfile as wav
import argparse
import json


import utils.spectrum_proc_utils as utils_sp
import utils.sig_utils as utils_sig
import utils.tdomain_proc_utils as utils_td
import utils.plot_utils as utils_plot
import utils.pitch_utils as utils_pitch
import utils.emph_detect_utils as utils_emph

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
    res['peak_to_peak_thr_std'] = 4.0
    res['out_dir'] = ''
    res['detect_hysteresis'] = 0.0001 # seconds
    res['detect_merge_threshold'] = 0.120
    res['detect_min_len'] = 0.02
    res['detect_max_len'] = 0.3
    res['debug_mode'] = False # enable/disable additional logging and s#@t
    res['spec_change_threshold'] = 3.0
    res['spec_change_band_st'] = 100 # hz
    res['spec_change_band_end'] = 3500  # hz
    res['detect_type'] = 1
    res['peak2change_thr'] = 1.3 # in some imaginary unknown logarithmic twisted units
    res['scan_region_len'] = 0.3 # seconds, length of window that we will position over detected regions and scan in the end

    return res

def dump_config(cfg, filename):
    cfgstr = json.dumps(cfg, indent = 2)
    # jsonobj = json.loads(cfgstr)
    with open(filename, 'w') as f:
        f.write(cfgstr)

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

    dtype = config['detect_type']

    if dtype == 1:
        return run_emp_detect_type1(wavfile, config, silent)
    elif dtype == 2:
        return run_emp_detect_type2(wavfile, config, silent)
    else:
        raise Exception('Unknown detection type specified!!! ({0})'.format(dtype))

def run_emp_detect_type1(wavfile, config, silent = True):

    # (DONE) 1) spectral change (basically envelope stability sort of)
    # (DONE) 2) peak-to-peak rate
    # (DONE) 3) syllable duration (basically a non-interrupted pitch segment)
    # (DONE) 4) pitch maxima (probably relatively to it's average value)

    (samplerate, signal) = wav.read(wavfile)
    signal = signal - numpy.mean(signal) # just in case, cause some inputs are really screwed
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

    # MY_DBG
    #utils_plot.plot_curves([signal], [signal_time])
    #input('eat a dick!')

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

    if (wrld_res[0] is None or wrld_res[1] is None or wrld_res[2] is None):
        raise Exception('LEFUCKUP')

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
    RESULT_MASK = update_detection_results(RESULT_MASK, samplerate, config['detect_hysteresis'],
                                           config['detect_merge_threshold'], config['detect_min_len'],
                                           config['detect_max_len'])

    # MY_DBG
    scan_segs = position_scan_regions(signal.squeeze(), RESULT_MASK, 513)
    #print(scan_segs)
    utils_plot.plot_emphasis_scan_segs(signal.squeeze(), RESULT_MASK, scan_segs, samplerate)
    input('eat some ass')

    # MY_DBG
    #utils_plot.plot_curves([signal], [signal_time])
    #utils_plot.plot_curves([signal, RESULT_MASK], [signal_time, signal_time])
    #input('eat a dick!')

    # one more time make sure unvoiced segs are not detected
    RESULT_MASK = RESULT_MASK * (DETECT_VO > 0)

    if not silent:
        utils_plot.plot_curves( [signal, RESULT_MASK], [signal_time, signal_time])

    SC_DFB = numpy.interp(signal_time.squeeze(), sc_time.squeeze(), sc_dfb.squeeze())
    P2P = numpy.interp(signal_time.squeeze(), sig_chunks_time.squeeze(), p2p.squeeze())

    dbg_stuff = {'spec_change_detect' : DETECT_SC, 'voiced_detect' : DETECT_VO, 'peak2peak_detect' : DETECT_PP,
        'f0-extreme_detect' : DETECT_EX, 'threshold_base' : std_no_sil, 'spec_change' : SC_DFB,
        'peak2peak' : P2P}

    return (RESULT_MASK, signal_time, dbg_stuff)

def run_emp_detect_type2(wavfile, config, silent = True):

    # ATTENTION!!! CURRENTLY NOT IMPLEMENTED!!!
    # need to work with
    # (DONE) 1) spectral change (basically envelope stability sort of)
    # (DONE) 2) peak-to-peak rate
    # (DONE) 3) syllable duration (basically a non-interrupted pitch segment)
    # (DONE) 4) pitch maxima (probably relatively to it's average value)

    (samplerate, signal) = wav.read(wavfile)
    signal = signal - numpy.mean(signal) # just in case, cause some inputs are really screwed
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

    # MY_DBG
    #utils_plot.plot_curves([signal, sc_res], [signal_time, sc_time])

    ### =================== PEAK-to_PEAK
    p2p = utils_td.get_peak_to_peak_from_chunks(sig_chunks)
    # MY_DBG
    #utils_plot.plot_curves( [signal, p2p], [signal_time, sig_chunks_time])

    ### =================== VOICED MASK
    voiced = (sig_f0.squeeze() > 0.0).astype('int')
    # MY_DBG
    #utils_plot.plot_curves( [sig_f0 / numpy.max(sig_f0), voiced], [sig_f0_time, sig_f0_time])

    # PEAK-TO-CHANGE
    p2p_int = numpy.interp(signal_time.squeeze(), sig_chunks_time.squeeze(), p2p.squeeze())
    sc_dfb_int = numpy.interp(signal_time.squeeze(), sc_time.squeeze(), sc_dfb.squeeze())
    p2sc = numpy.log(p2p_int) - numpy.log(sc_dfb_int)
    p2sc = utils_sig.clean_undef_floats(p2sc)

    ### =================== FINALIZING RESULTS
    DETECT_VO = numpy.interp(signal_time.squeeze(), sig_f0_time.squeeze(), voiced.squeeze())
    DETECT_EX = numpy.interp(signal_time.squeeze(), sig_f0_time.squeeze(), f0_extr.squeeze())
    DETECT_P2SC = p2sc > config['peak2change_thr']

    RESULT_MASK = (DETECT_P2SC > 0) * (DETECT_VO > 0) * (DETECT_EX > 0)
    #RESULT_MASK = update_detection_results(RESULT_MASK, samplerate, config['detect_hysteresis'],
    #                                       config['detect_merge_threshold'])
    # one more time make sure unvoiced segs are not detected
    #RESULT_MASK = RESULT_MASK * (DETECT_VO > 0)

    if not silent:
        utils_plot.plot_curves( [signal, RESULT_MASK], [signal_time, signal_time])

    SC_DFB = numpy.interp(signal_time.squeeze(), sc_time.squeeze(), sc_dfb.squeeze())
    P2P = numpy.interp(signal_time.squeeze(), sig_chunks_time.squeeze(), p2p.squeeze())

    dbg_stuff = {'peak2schange_detect' : DETECT_P2SC, 'voiced_detect' : DETECT_VO,
        'f0-extreme_detect' : DETECT_EX, 'threshold_base' : std_no_sil, 'spec_change' : SC_DFB,
        'peak2peak' : P2P, 'peak2schange' : p2sc}

    return (RESULT_MASK, signal_time, dbg_stuff)

def update_detection_results(mask, samplerate, detect_hysteresis, merge_threshold, min_len, max_len):
    assert(utils_sig.is_array(mask))
    hyst_step = int(numpy.round(detect_hysteresis * samplerate))
    merge_step = int(numpy.round(merge_threshold * samplerate))
    min_keep_len = int(numpy.round(min_len * samplerate))
    max_keep_len = int(numpy.round(max_len * samplerate))
    # MY_DBG
    print("hyst_step = {0}  merge_step = {1} ...".format(hyst_step, merge_step))
    sig_len = len(mask)
    idx = 0
    result = numpy.copy(mask)
    while(idx < sig_len):
        if (mask[idx] > 0.0):
            result[idx : idx + hyst_step] = 1.0
            idx += hyst_step
            continue
        idx += 1

    detect_segs = segs_list_from_signal(result)
    detect_segs = merge_segs(detect_segs, min_len=min_keep_len, max_len = max_keep_len,
                             merge_thr=merge_step)
    result = segs_list_to_signal(detect_segs, len(result))

    return result

def segs_list_from_signal(sig):
    idx = 0
    assert(utils_sig.is_array(sig))
    res = []
    while (idx < len(sig)):
        if (sig[idx] > 0):
            st = idx
            while(sig[idx] > 0):
                idx += 1
            end = idx
            res.append( {'st':st, 'end':end} )
            continue
        idx += 1
    return res

def merge_segs(segs, min_len, max_len, merge_thr):

    def should_delete(seg):
        return (seg['end'] - seg['st']) <= min_len

    def can_merge(lst, idx1, idx2):
        assert idx1 != idx2
        c1 = lst[idx1]['end'] - lst[idx1]['st'] < max_len
        c2 = lst[idx2]['end'] - lst[idx2]['st'] < max_len
        if idx1 > idx2:
            c3 = (lst[idx1]['st'] - lst[idx2]['end']) < merge_thr
            return (c1 and c2 and c3)
        else:
            c3 = (lst[idx2]['st'] - lst[idx1]['end']) < merge_thr
            return (c1 and c2 and c3)

    def merge_segs(lst, from_which, to_which):
        assert(from_which != to_which)
        lst[to_which]['segs'] += lst[from_which]['segs']
        lst[to_which]['st'] = min(lst[to_which]['st'], lst[from_which]['st'])
        lst[to_which]['end'] = max(lst[to_which]['end'], lst[from_which]['end'])
        lst[from_which]['segs'] = []

    wrk_segs = []
    k = 0
    for seg in segs:
        wrk_segs.append( {'st':seg['st'], 'end':seg['end'], 'segs':[k]} )
        k += 1

    for k in range(len(wrk_segs)):

        # handle last
        if k == (len(wrk_segs) - 1):
            if can_merge(wrk_segs, idx1 = k, idx2 = k - 1):
                merge_segs(wrk_segs, from_which = k, to_which = k - 1)
            else:
                if should_delete(wrk_segs[k]):
                    wrk_segs[k]['segs'] = []
            continue

        # handle the rest
        if can_merge(wrk_segs, idx1 = k, idx2 = k + 1):
            merge_segs(wrk_segs, from_which = k, to_which = k + 1)
        else:
            if should_delete(wrk_segs[k]):
                wrk_segs[k]['segs'] = []

    res = []
    for seg in wrk_segs:
        if seg['segs']:
            res.append(seg)

    return res

def segs_list_to_signal(segs, total_len):
    res = numpy.zeros(total_len)
    for seg in segs:
        res[seg['st']:seg['end']] = 1
    return res

def position_scan_regions(sig, detect_mask, scan_len):
    detect_segments = segs_list_from_signal(detect_mask)
    assert(utils_sig.is_array(sig))
    assert(utils_sig.is_array(detect_mask))
    assert((scan_len % 2) == 1)

    half_seg_len = (scan_len - 1)/2
    fsig = utils_emph.condition_signal_for_emph_scanning(sig)
    res = []
    for dseg in detect_segments:
        dchunk = fsig[dseg['st']:dseg['end']]
        max_idx = numpy.argmax(dchunk)
        #assert(len(max_idx) == 1) # check that we have only 1 maximum value in the result, otherwise it is VERY strange
        scan_set_st = int(dseg['st'] + max_idx - half_seg_len)
        scan_set_end = int(dseg['st'] + max_idx + half_seg_len)
        res.append({'st':scan_set_st, 'end':scan_set_end})

    return res

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

    if 'spec_change' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_spec_change.png'
        thr_y = CFG['spec_change_threshold'] * threshold_base * numpy.ones( len(signal_time))
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['spec_change'], thr_y],
                           base_plot_x + [signal_time, signal_time],
                           labels=base_plot_labels + ['spec_change', 'threshold'],
                           saveto=plotname)

    if 'voiced_detect' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_voiced.png'
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['voiced_detect']],
                           base_plot_x + [signal_time], labels=base_plot_labels + ['voiced'],
                           saveto=plotname)

    if 'peak2peak' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_p2p.png'
        thr_y = CFG['peak_to_peak_thr_std'] * threshold_base * numpy.ones(len(signal_time))
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['peak2peak'], thr_y],
                           base_plot_x + [signal_time, signal_time],
                           labels=base_plot_labels + ['peak2peak', 'threshold'],
                           saveto=plotname)

    if 'f0-extreme_detect' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_f0_extr.png'
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['f0-extreme_detect']],
                           base_plot_x + [signal_time], labels=base_plot_labels + ['f0-extreme'],
                           saveto=plotname)

    if 'peak2schange_detect' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_p2sc_detect.png'
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['peak2schange_detect']],
                           base_plot_x + [signal_time], labels=base_plot_labels + ['peak2schange_detect'],
                           saveto=plotname)

    if 'peak2schange' in dbg_stuff.keys():
        plotname = os.path.splitext(outname)[0] + '_p2sc.png'
        thr_y2 = CFG['peak2change_thr'] * threshold_base * numpy.ones(len(signal_time))
        utils_plot.plot_curves(base_plot_y + [dbg_stuff['peak2schange'], thr_y2],
                           base_plot_x + [signal_time, signal_time],
                           labels=base_plot_labels + ['peak2schange', 'threshold'],
                           saveto=plotname)

if __name__ == '__main__':

    ARGS = parse_input_args()

    mode = ARGS.mode

    input_data = ARGS.i
    output_data = ARGS.o
    mask_data = ARGS.m

    # Go on from 'MY_DBG' point and debug the scan-segments returned by implemented routine

    mode = ARGS.mode
    if mode is None:
        mode = 'file'

    if mode == 'dir':
        if not os.path.exists(output_data):
            os.makedirs(output_data)

    if ARGS.cfg is None:
        CFG = get_default_config()
        cfg_filename = os.path.join(output_data, 'used_config.cfg') if mode == 'dir' else 'used_config.cfg'
        dump_config(CFG, cfg_filename)
    else:
        CFG = get_config_from_json(ARGS.cfg)


    if mode == 'file':
        # run everything to process one input file only
        run_main_one_file(input_data, output_data, mask_data, CFG)
    elif mode == 'dir':
        # run everything to process a set of files in specified dir
        run_main_proc_dir(input_data, output_data, mask_data, CFG)
    else:
        raise Exception('ERROR : some unknown mode scecified')

