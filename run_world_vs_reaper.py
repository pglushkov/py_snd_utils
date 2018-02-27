
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

def run_main(wav_name, out_path, timestep):

    REAPER_PATH = './bin_utils/reaper'
    WORLD_PATH = './bin_utils/world_analysis'

    (samplerate, wav_signal) = wav.read(wav_name)
    sampleperiod = 1.0 / samplerate
    wav_signal = wav_signal / (2.0 ** 15)

    reaper_f0_file, _ = utils_reaper.run_reaper(wav_name, REAPER_PATH, out_path)
    world_f0_file, _, _ = utils_world.run_world(wav_name, WORLD_PATH, out_path)

    reaper_time, reaper_f0 = utils_reaper.read_f0_from_file(reaper_f0_file)
    world_f0 = numpy.fromfile(world_f0_file)

    assert(reaper_time[1] - reaper_time[0] == timestep)
    world_time = numpy.arange(len(world_f0)) * timestep
    sig_time = numpy.arange(len(wav_signal)) * sampleperiod

    utils_plot.plot_curves( [reaper_f0, world_f0, wav_signal * 200.0], x = [reaper_time, world_time, sig_time],
                            labels = ['reaper', 'world', 'signal'])

    #print(reaper_res)
    #print(world_res)

if __name__ == '__main__':

    TIMESTEP = 0.005

    if len(sys.argv) < 2:
        raise Exception('Need to specify at least input wav file!')

    wav_name = sys.argv[1]

    if (len(sys.argv) >= 3):
        out_path = sys.argv[2]
    else:
        out_path = './'

    run_main(wav_name, out_path, TIMESTEP)
