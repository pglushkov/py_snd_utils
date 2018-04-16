
import numpy
import scipy.signal as sci_sig

def condition_signal_for_emph_scanning(sig):
    # for now the only conditioning is LP-filtering the signal to remove jumpyness
    fir = sci_sig.hann(65)
    res = sci_sig.filtfilt(fir, 1, sig)
    return res