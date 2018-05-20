
import os
import sys
import numpy
import scipy.signal as sci_sig
import scipy.io.wavfile as wav
import argparse
import json


import utils.plot_utils as utils_plot
import utils.emph_detect_utils as utils_emph


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

    (detect, detect_time, scan_segs, dbg_stuff) = \
        utils_emph.run_emp_detect(wavname, CFG, silent = True)

    (samplerate, signal) = wav.read(wavname)
    signal_time = numpy.arange(len(signal)) / samplerate

    #plot_curves_y = [signal/(2.0 ** 15.0), detect]
    #plot_curves_x = [signal_time, detect_time]
    #plot_labels = ['input signal', 'detection mask']

    plot_curves_y = [signal/(2.0 ** 15.0)]
    plot_curves_x = [signal_time]
    plot_labels = ['input signal']

    if maskfile is not None:
        (samplerate, mask) = wav.read(maskfile)
        mask = mask / (2.0 ** 15.0)
        mask_time = numpy.arange(len(mask)) / samplerate
        plot_curves_y.append(mask)
        plot_curves_x.append(mask_time)
        plot_labels.append('reference manual mask')

    if scan_segs is not None:
        (seg_x, seg_y) = utils_plot.get_scan_seg_plots(scan_segs, samplerate, len(signal))
        seg_num = 1
        for x, y in zip(seg_x, seg_y):
            plot_curves_y.append(y)
            plot_curves_x.append(x)
            plot_labels.append('scan seg {0}'.format(seg_num))
            seg_num += 1

    detected_signal = signal * detect

    print('All done ...')
    print('Saving WAV result to {0}'.format(outname))
    wav.write(outname, samplerate, detected_signal)

    plotname = os.path.splitext(outname)[0] + '.png'
    utils_plot.plot_curves(plot_curves_y, plot_curves_x, labels = plot_labels, saveto = plotname)
    print('Saving plot to {0}'.format(plotname))

    scan_segs_descr = utils_emph.scan_segs_to_json(scan_segs, samplerate)
    jsonname = os.path.splitext(outname)[0] + '.json'
    with open(jsonname, 'w') as of:
        of.write(json.dumps(scan_segs_descr))

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

    mode = ARGS.mode
    if mode is None:
        mode = 'file'

    if mode == 'dir':
        if not os.path.exists(output_data):
            os.makedirs(output_data)

    if ARGS.cfg is None:
        CFG = utils_emph.get_default_config()
        cfg_filename = os.path.join(output_data, 'emp_detect_used_config.cfg') if mode == 'dir' else 'used_config.cfg'
        dump_config(CFG, cfg_filename)
    else:
        CFG = get_config_from_json(ARGS.cfg)

    if not os.path.exists(CFG['wrk_path']):
        print('Specified working directory does not exist, creating one')
        os.makedirs(CFG['wrk_path'])

    if mode == 'file':
        # run everything to process one input file only
        run_main_one_file(input_data, output_data, mask_data, CFG)
    elif mode == 'dir':
        # run everything to process a set of files in specified dir
        run_main_proc_dir(input_data, output_data, mask_data, CFG)
    else:
        raise Exception('ERROR : some unknown mode scecified')

