

import wave
import numpy
import os
import shutil
import sys
import subprocess

from scipy import signal as sig
from matplotlib import pyplot as plt

def run_proc(procname, args):
    argstr = [procname]
    argstr += args    
    return subprocess.call(argstr)

def read_wav_from_file(wav_name):
    with wave.open(wav_name) as WAV:
        samples = WAV.readframes(WAV.getnframes())
        params = WAV.getparams()
    
        frameperiod = float(1.0 / params.framerate)

        out_samples = numpy.frombuffer(samples, dtype='int16').reshape( (-1,1) )
        time_grid = numpy.arange(0.0, frameperiod * params.nframes, frameperiod)

    return (out_samples, time_grid, params)

def read_f0_from_file(f0_name):
    with open(f0_name, 'r') as F:
        strs = F.readlines()

    strs = strs[7:-1]
    numsamples = len(strs)

    f0 = numpy.zeros( (numsamples, 1) )
    times = numpy.zeros( (numsamples, 1) )
    voiced = numpy.zeros( (numsamples, 1) )
    
    for k in range(numsamples):
        (times[k,0], voiced[k,0], f0[k,0]) = strs[k].split(' ')

    return (times, f0)

def read_pm_from_file(pm_name):
    with open(pm_name, 'r') as F:
        strs = F.readlines()

    strs = strs[7:-1]
    numsamples = len(strs)

    times = numpy.zeros( (numsamples, 1) )
    voiced = numpy.zeros( (numsamples, 1) )
    
    for k in range(numsamples):
        (times[k,0], voiced[k,0], _) = strs[k].split(' ')

    return (times, voiced)


def chunk_signal_by_pm(signal, marks, samplerate, fft_size):
    idxs = (marks * float(samplerate)).astype('int32')
    reslen = idxs.shape[0]
    res = numpy.zeros( (reslen, fft_size) )
   
    #MY_DBG
    #print(idxs)

    st_idx = 0
    for k in range(reslen):
        end_idx = idxs[k,0]
        
        #MY_DBG
        #print('     will cut from {0} to {0} ...\n'.format(st_idx, end_idx))

        res[k, 0 : (end_idx - st_idx)] = signal[st_idx : end_idx, 0]
        st_idx = end_idx

    return res

def run_main():

    if len(sys.argv) < 2:
        raise Exception("no WAV input name is given!")

    wav_name = sys.argv[1]
    f0_name = os.path.splitext(os.path.basename(wav_name))[0] + '_f0.txt'
    pm_name = os.path.splitext(os.path.basename(wav_name))[0] + '_pm.txt'
    
    curpath = os.getcwd()

    reaper_name = curpath + '/reaper'
    reaper_args = ['-i', curpath + '/' + wav_name, '-f', curpath + '/' + f0_name, '-p',  curpath + '/' + pm_name, '-a']

    reaper_res = run_proc(reaper_name, reaper_args)
    print("Running REAPER returned result : {0}".format(reaper_res))

    if reaper_res != 0:
        raise Exception("REAPER returned error result = {0}".format(reaper_res))

    wav_track = read_wav_from_file(wav_name)
    samplerate = wav_track[2].framerate

    #MY_DBG
    #print(len(wav_track[0]))
    #print(len(wav_track[1]))
    #print(wav_track[2])
    #plt.figure()
    #plt.plot(wav_track[1], wav_track[0])
    #plt.show()
    #return
    
    f0_track = read_f0_from_file(f0_name)

    #MY_DBG
    #print(len(f0_track[0])) 
    #plt.figure()
    #plt.plot(f0_track[0], f0_track[1])
    #plt.show()

    pm_track = read_pm_from_file(pm_name)
         
    #MY_DBG
    print(len(pm_track[0]))
    #plt.figure()
    #plt.plot( numpy.diff(pm_track[0]))
    #plt.plot((pm_track[0]))
    #plt.show()
    #some_cock = numpy.diff(pm_track[0], axis = 0)
    #print(some_cock)
    #plt.figure()
    #plt.plot(some_cock)
    #plt.show()

    pm_intervs = numpy.diff(pm_track[0], axis = 0)
    fft_size = int(2.0 ** numpy.ceil(numpy.log2(numpy.max(pm_intervs * samplerate))))
    #print(fft_size)
    wnd = sig.get_window('boxcar', fft_size)
  

  
    sig_for_fft = chunk_signal_by_pm(wav_track[0], pm_track[0], samplerate, fft_size)
    #print(sig_for_fft)

    #MY_DBG
    #plt.figure()
    #plt.plot(sig_for_fft.flatten(0))
    #plt.show()

    sig_for_sgram = sig_for_fft.flatten(0)
    #freq_grid, time_grid, sgram = sig.spectrogram(sig_for_sgram, fs=samplerate, window=wnd, nperseg=fft_size, noverlap=0, 
    #        nfft=fft_size, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')

    #plt.pcolormesh(time_grid, freq_grid, sgram)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()


    sig_for_sgram.astype('float32').tofile('pmarked.bin')
    wav_track[0].squeeze().astype('float32').tofile('orig.bin')

    freq_grid, time_grid, sgram = sig.spectrogram(wav_track[0].squeeze(), fs=samplerate)

    plt.pcolormesh(time_grid, freq_grid, sgram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



if __name__ == '__main__':
    run_main()





