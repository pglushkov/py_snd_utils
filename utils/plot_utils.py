
import numpy
import matplotlib.pyplot as plt


def simple_plot(y, x = None):
    
    if x is None:
        x = numpy.arange(y.shape[0])

    f = plt.figure()
    plt.plot(x,y)
    f.show()
    input('press ENTER to go on ...')

def plot_curves(y, x = None):
    assert(y.__class__ is list)
    if x is not None:
        assert(x.__class__ is list)
    assert (len(y) == len(x))

    f = plt.figure()
    for k in range(len(y)):
        Y = y[k].squeeze()
        if x is None:
            X = numpy.arange(len(Y))
        else:
            X = x[k].squeeze()
        plt.plot(X,Y)
    f.show()
    input('press ENTER to go on...')
