
import numpy
import os
import sys
import scipy.io.wavfile as wav

import utils.misc_utils as utils_misc
import utils.reaper_utils as utils_reaper

def run_world(wav_name, world_name, out_path = '', inf0 = None):
    f0_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.f0'
    sp_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.sp'
    ap_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.ap'

    f0_file_name = os.path.join(out_path, f0_name)
    sp_file_name = os.path.join(out_path, sp_name)
    ap_file_name = os.path.join(out_path, ap_name)
    world_args = ['--in', wav_name, '--f0', f0_file_name, '--sp', sp_file_name, '--ap', ap_file_name]

    if inf0 is not None:
        assert(os.path.exists(inf0))
        world_args += [ '--inf0', inf0]

    reaper_res = utils_misc.run_process(world_name, world_args)
    print("Running WORLD returned result : {0}".format(reaper_res))

    if reaper_res != 0:
        raise Exception("WORLD returned error result = {0}".format(reaper_res))

    return (f0_file_name, sp_file_name, ap_file_name)


def run_world_by_reaper(wav_name, out_path, rpath = None, wpath = None):

    if sys.platform == 'win32':
        # world and reaper are currently not supplied in binary form for Windows,
        # will try to see if needed files had already been created previously. If not -
        # return 'None' values
        return try_search_previous_results(wav_name, out_path)

    REAPER_PATH = './bin_utils/reaper' if rpath is None else rpath
    WORLD_PATH = './bin_utils/world_analysis' if wpath is None else wpath

    (samplerate, wav_signal) = wav.read(wav_name)
    sampleperiod = 1.0 / samplerate
    wav_signal = wav_signal / (2.0 ** 15)

    reaper_f0_file, _ = utils_reaper.run_reaper(wav_name, REAPER_PATH, out_path)
    reaper_time, reaper_f0 = utils_reaper.read_f0_from_file(reaper_f0_file)
    reaper_f0_bin_file = os.path.splitext(reaper_f0_file)[0] + '.f0'
    reaper_f0.tofile(reaper_f0_bin_file)

    wrld_result = \
        run_world(wav_name, WORLD_PATH, out_path, inf0 = reaper_f0_bin_file)

    print("Results are : {0}, {1}, {2}".format(wrld_result[0], wrld_result[1], wrld_result[2]))

    return wrld_result

def try_search_previous_results(wav_name, out_path):
    fid = os.path.splitext(os.path.basename(wav_name))[0]

    f0_fname = fid + '_world.f0'
    ap_fname = fid + '_world.ap'
    sp_fname = fid + '_world.sp'

    if os.path.isfile(f0_fname):
        print("ERROR : could not get WORLD data for file {0}...".format(f0_fname))
        f0_fname = None
    else:
        f0_fname = os.path.join(out_path, f0_fname)

    if os.path.isfile(ap_fname):
        print("ERROR : could not get WORLD data for file {0}...".format(ap_fname))
        ap_fname = None
    else:
        ap_fname = os.path.join(out_path, ap_fname)

    if os.path.isfile(sp_fname):
        print("ERROR : could not get WORLD data for file {0}...".format(sp_fname))
        sp_fname = None
    else:
        sp_fname = os.path.join(out_path, sp_fname)

    return (f0_fname, sp_fname, ap_fname)