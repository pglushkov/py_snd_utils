
import numpy
import scipy.signal as sci_sig

def condition_signal_for_emph_scanning(sig):
    # for now the only conditioning is LP-filtering the signal to remove jumpyness
    fir = sci_sig.hann(401)
    fir /= numpy.sum(fir)
    #fir = sci_sig.firwin(400, 0.02)
    res = sci_sig.convolve(numpy.abs(sig), fir, 'same')
    return res