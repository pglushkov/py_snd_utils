
import numpy
import subprocess


def nextpow2(inp):
    return int ( 2.0 ** numpy.ceil(numpy.log2(inp)) )

def run_process(procname, args):
    argstr = [procname]
    argstr += args
    return subprocess.call(argstr)



