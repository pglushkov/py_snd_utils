
import numpy
import os

import utils.misc_utils as utils_misc

def run_world(wav_name, world_name, out_path = '', inf0 = None):
    f0_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.f0'
    sp_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.sp'
    ap_name = os.path.splitext(os.path.basename(wav_name))[0] + '_world.ap'

    f0_file_name = os.path.join(out_path, f0_name)
    sp_file_name = os.path.join(out_path, sp_name)
    ap_file_name = os.path.join(out_path, ap_name)
    world_args = ['--in', wav_name, '--f0', f0_file_name, '--sp', sp_file_name, '--ap', ap_file_name]

    if inf0 is not None:
        assert(os.path.exists(inf0))
        world_args += [ '--inf0', inf0]

    reaper_res = utils_misc.run_process(world_name, world_args)
    print("Running WORLD returned result : {0}".format(reaper_res))

    if reaper_res != 0:
        raise Exception("WORLD returned error result = {0}".format(reaper_res))

    return (f0_file_name, sp_file_name, ap_file_name)



