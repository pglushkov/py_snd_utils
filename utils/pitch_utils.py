
import numpy

import utils.sig_utils as utils_sig

def get_f0_extreme_areas(inf0, thr = 0.5, seg_len_thr = 10):

    assert (utils_sig.is_vector(inf0))

    tmp_f0 = numpy.squeeze(inf0)

    f0 = tmp_f0[tmp_f0 > 0.0]
    f0_mean = numpy.mean(f0)
    f0_std = numpy.std(f0)

    high_mask = (tmp_f0 > f0_mean + f0_std * thr).astype('float64')
    low_mask = (tmp_f0 > 0.0).astype('float64') * (tmp_f0 < f0_mean - f0_std * thr).astype('float64')

    high_mask = select_good_extreme_segments(high_mask, seg_len_thr)
    low_mask = select_good_extreme_segments(low_mask, seg_len_thr)

    f0_high = inf0 * high_mask.reshape(inf0.shape)
    f0_low = inf0 * low_mask.reshape(inf0.shape)

    return (f0_low, f0_high)


def select_good_extreme_segments(inp, thr):
    result = numpy.zeros( inp.shape ).squeeze()
    tmp = numpy.squeeze(inp)
    k = 0
    while k < len(tmp):
        if tmp[k] > 0:
            m = k + 1
            while tmp[m] > 0:
                m += 1
            cur_seg = (k,m)
            if cur_seg[1] - cur_seg[0] > thr:
                result[cur_seg[0]:cur_seg[1]] = tmp[cur_seg[0]:cur_seg[1]]
                # MY_DBG
                #print('Taking from {0} to {1}'.format(cur_seg[0], cur_seg[1]))
            k = m
            continue
        k += 1
    return result.reshape( inp.shape )

