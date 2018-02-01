
import numpy

def nextpow2(inp):
    return int ( 2.0 ** numpy.ceil(numpy.log2(inp)) )