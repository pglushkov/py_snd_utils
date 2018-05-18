
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


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception('Need to specify at least input wav file!')

    wav_name = sys.argv[1]

    if (len(sys.argv) >= 3):
        out_path = sys.argv[2]
    else:
        out_path = './'

    run_world_by_reaper(wav_name, out_path)
