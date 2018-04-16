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


def plot_emphasis_scan_segs(signal, detect, scan_segs, samplerate):
    assert (utils_sig.is_array(signal))
    assert (utils_sig.is_array(detect))

    sig_time = numpy.arange(len(signal)) * samplerate
    # sig_cond = utils_emph.condition_signal_for_emph_scanning(signal)
    # plt_y = [signal, sig_cond, detect]
    # plt_x = [sig_time, sig_time, sig_time]
    plt_y = [signal, detect]
    plt_x = [sig_time, sig_time]
    for scan_seg in scan_segs:
        tmpy = numpy.zeros(len(signal))
        tmpy[scan_seg['st']:scan_seg['end']] = 1.0
        plt_y.append(tmpy)
        plt_x.append(sig_time)

    return plot_curves(plt_y, plt_x)