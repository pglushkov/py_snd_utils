
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
import utils.world_utils as utils_world

def run_world_by_reaper(wav_name, out_path, rpath = None, wpath = None):

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
        utils_world.run_world(wav_name, WORLD_PATH, out_path, inf0 = reaper_f0_bin_file)

    print("Results are : {0}, {1}, {2}".format(wrld_result[0], wrld_result[1], wrld_result[2]))

    return wrld_result

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception('Need to specify at least input wav file!')

    wav_name = sys.argv[1]

    if (len(sys.argv) >= 3):
        out_path = sys.argv[2]
    else:
        out_path = './'

    run_world_by_reaper(wav_name, out_path)
