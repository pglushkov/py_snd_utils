
import numpy
import matplotlib.pyplot as plt


def simple_plot(y, x = None):
    
    if x is None:
        x = numpy.arange(y.shape[0])

    f = plt.figure()
    plt.plot(x,y)
    f.show()
    input('press ENTER to go on ...')

def plot_curves(y, x = None, labels = None):
    assert(y.__class__ is list)
    if x is not None:
        assert(x.__class__ is list)
        assert (len(y) == len(x))
    if labels is not None:
        assert(labels.__class__ is list)
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
        plt.plot(X,Y, label = lab)
    plt.legend()
    f.show()
    input('press ENTER to go on...')
