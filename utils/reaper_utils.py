
import numpy
import os

import utils.misc_utils as utils_misc

def read_f0_from_file(f0_name):
    with open(f0_name, 'r') as F:
        strs = F.readlines()

    strs = strs[7:-1]
    numsamples = len(strs)

    f0 = numpy.zeros((numsamples, 1))
    times = numpy.zeros((numsamples, 1))
    voiced = numpy.zeros((numsamples, 1))

    for k in range(numsamples):
        (times[k, 0], voiced[k, 0], f0[k, 0]) = strs[k].split(' ')

    return (times, f0)


def read_pm_from_file(pm_name):
    with open(pm_name, 'r') as F:
        strs = F.readlines()

    strs = strs[7:-1]
    numsamples = len(strs)

    times = numpy.zeros((numsamples, 1))
    voiced = numpy.zeros((numsamples, 1))

    for k in range(numsamples):
        (times[k, 0], voiced[k, 0], _) = strs[k].split(' ')

    return (times, voiced)


def chunk_signal_by_pm(signal, marks, samplerate, fft_size):
    idxs = (marks * float(samplerate)).astype('int32')
    reslen = idxs.shape[0]
    res = numpy.zeros((reslen, fft_size))

    # MY_DBG
    # print(idxs)

    st_idx = 0
    for k in range(reslen):
        end_idx = idxs[k, 0]

        # MY_DBG
        # print('     will cut from {0} to {0} ...\n'.format(st_idx, end_idx))

        res[k, 0: (end_idx - st_idx)] = signal[st_idx: end_idx, 0]
        st_idx = end_idx

    return res

def run_reaper(wav_name, reaper_name, out_path = ''):
    f0_name = os.path.splitext(os.path.basename(wav_name))[0] + '_reaper_f0.txt'
    pm_name = os.path.splitext(os.path.basename(wav_name))[0] + '_reaper_pm.txt'

    f0_file_name = os.path.join(out_path, f0_name)
    pm_file_name = os.path.join(out_path, pm_name)
    reaper_args = ['-i', wav_name, '-f', f0_file_name, '-p', pm_file_name, '-a']

    reaper_res = utils_misc.run_process(reaper_name, reaper_args)
    print("Running REAPER returned result : {0}".format(reaper_res))

    if reaper_res != 0:
        raise Exception("REAPER returned error result = {0}".format(reaper_res))

    return (f0_file_name, pm_file_name)