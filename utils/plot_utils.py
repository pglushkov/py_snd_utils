
import numpy
import matplotlib.pyplot as plt


def simple_plot(y, x = None):
    
    if x is None:
        x = numpy.arange(y.shape[0])

    f = plt.figure()
    plt.plot(x,y)
    f.show()
    input('press ENTER to go on ...')
