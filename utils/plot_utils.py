import numpy
import matplotlib.pyplot as plt

import utils.emph_detect_utils as utils_emph
import utils.sig_utils as utils_sig


def simple_plot(y, x=None):
    if x is None:
        x = numpy.arange(y.shape[0])

    f = plt.figure()
    plt.plot(x, y)
    f.show()
    input('press ENTER to go on ...')


def plot_curves(y, x=None, labels=None, saveto=None):
    assert (y.__class__ is list)
    if x is not None:
        assert (x.__class__ is list)
        assert (len(y) == len(x))
    if labels is not None:
        assert (labels.__class__ is list)
        assert (len(y) == len(labels))

    f = plt.figure()
    for k in range(len(y)):
        Y = y[k].squeeze()
        if x is None:
            X = numpy.arange(len(Y))
        else:
            X = x[k].squeeze()
        if labels is None:
            lab = 'plot' + str(k)
        else:
            lab = labels[k]
        plt.plot(X, Y, label=lab)
    plt.legend()

    if (saveto is not None):
        plt.savefig(saveto)
    else:
        f.show()
        input('press ENTER to go on...')


def get_scan_seg_plots(scan_segs, samplerate, sig_len):
    plot_seg_y = []
    plot_seg_x = []
    xaxis = numpy.arange(sig_len) / samplerate;
    for seg in scan_segs:
        tmpy = numpy.zeros(sig_len)
        tmpy[seg['st']:seg['end']] = 1.0
        plot_seg_y.append(tmpy)
        plot_seg_x.append(xaxis)
    return (plot_seg_x, plot_seg_y)

def plot_emphasis_scan_segs(signal, detect, scan_segs, samplerate, filename = None):
    assert (utils_sig.is_array(signal))
    assert (utils_sig.is_array(detect))

    sig_time = numpy.arange(len(signal)) / samplerate
    sig_cond = utils_emph.condition_signal_for_emph_scanning(signal)
    plt_y = [signal, sig_cond, detect]
    plt_x = [sig_time, sig_time, sig_time]
    (seg_x, seg_y) = get_scan_seg_plots(scan_segs, samplerate, len(signal))

    plt_y = plt_y + seg_y
    plt_x = plt_x + seg_x

    return plot_curves(plt_y, plt_x, saveto = filename)