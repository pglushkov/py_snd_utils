
# Small set of utility-functions to perform basic manupulations with the
# spectal-envelope of the signal

import numpy
import collections

# accepts scalar or numpy array as input
def mel2freq(mel):
    return 700.0*(10.0**(mel/2595.0) - 1.0)

# accepts scalar or numpy array as input
def freq2mel(freq):
    return 2595.0 * numpy.log10(1.0 + freq / 700.0)

def get_uniform_mel_freq_grid(freq_brd, num_mels):
    mel_end = freq2mel(freq_brd)
    mel_grid = numpy.linspace(0, mel_end, num_mels);
    mel_freq_grid = mel2freq(mel_grid)
    return mel_freq_grid

# takes spectral envelope acquired by FFT, samplerate of the signal
# and return 'resampled' envelope with new frequency point spaced uniformly
# in mels space
def interp_spec_in_mels_uniform(spectrum, fs, num_mels):
    nyquist = fs / 2
    mel_freq_grid = get_uniform_mel_freq_grid(nyquist, num_mels)
    # if spectrum was acquired by FFT then it spans from 0 to nyquist frequency
    orig_freq_grid = numpy.linspace(0, nyquist, len(spectrum))

    # had this issue in Octave, just in case it won't hurt to check
    if mel_freq_grid[-1] != nyquist:
        mel_freq_grid[-1] = nyquist
    if orig_freq_grid[-1] != nyquist:
        orig_freq_grid[-1] = nyquist

    mel_spectrum = numpy.interp(mel_freq_grid, orig_freq_grid, spectrum)
    return (mel_freq_grid, mel_spectrum)

# here we expect input mel-spectrum to cover interval from 0 to fs/2 Hz
# uniformly in mels space
def interp_spec_in_freq_uniform(mel_spectrum, fs, num_freqs):
    nyquist = fs /2
    freq_grid = numpy.linspace(0, nyquist, num_freqs)
    mel_freq_grid = get_uniform_mel_freq_grid(nyquist, len(mel_spectrum))

    # had this issue in Octave, just in case it won't hurt to check
    if mel_freq_grid[-1] != nyquist:
        mel_freq_grid[-1] = nyquist
    if freq_grid[-1] != nyquist:
        freq_grid[-1] = nyquist

    spectrum = numpy.interp(freq_grid, mel_freq_grid, mel_spectrum)
    return (freq_grid, spectrum)

# expecting data to be a linear array which needs to be reshaped
def transform_sp_data_to_mels(sp_data, fs, spec_dim, mel_dim):
    data = numpy.reshape(sp_data, (-1, spec_dim))
    new_data = numpy.zeros((data.shape[0], mel_dim), dtype='float32')
    for idx in range(data.shape[0]):
        mel_freq_bins, new_data[idx,:] = interp_spec_in_mels_uniform(data[idx,:], fs, mel_dim)
    return new_data

# expecting data to be a linear array which needs to be reshaped
def transform_mel_data_to_sp(mel_data, fs, mel_dim, sp_dim):
    data = numpy.reshape(mel_data, (-1, mel_dim))
    new_data = numpy.zeros((data.shape[0], sp_dim), dtype='float32')
    for idx in range(data.shape[0]):
        sp_freq_bins, new_data[idx,:] = interp_spec_in_freq_uniform(data[idx,:], fs, sp_dim)
    return new_data

def transform_lin_to_log(data, threshold = None):
    # for now adding small noise to the data to avoid Nan's and extremely small values
    if threshold is not None:
        # expect 'threshold' to be a power-dB value
        data += transform_log_to_lin(threshold)
    
    tmp = 20.0*numpy.log10(data)
    # MY_DBG
    if threshold is not None and numpy.sum(tmp < threshold) > 0:
        print("ERRORS WITH LIN-TO-LOG !!!");
        raise Exception('Unexpected result of transfrom_lin_to_log() operation!')

    return tmp
        

def transform_log_to_lin(data):
    return 10.0**(data/20.0)

def get_filter_bank_boarders(fmin, fmax, n_filters):

    n_mid_filters = n_filters - 2;

    n_centers = n_mid_filters + 2; # including start/end boarders
    n_bands = n_mid_filters + 2; # againg including start/end boarders, they are crucial for interpolation

    melmin = freq2mel(fmin);
    melmax = freq2mel(fmax);

    # OCTAVE version : mel_brds = melmin : (melmax - melmin) / n_mid_filters : melmax;
    mel_brds = numpy.arange(melmin, melmax, (melmax - melmin) / n_mid_filters)
    
    # F@#K Numpy !!!
    if (len(mel_brds) != n_mid_filters + 1):
        mel_brds = numpy.append(mel_brds, [melmax], 0)
    
    freq_brds = mel_brds;
  
    freq_centers = numpy.zeros(n_bands);
    freq_centers[0] = freq_brds[0];
    freq_centers[-1] = freq_brds[-1];
    
    for k in range(1, len(freq_centers) - 1, 1):
        freq_centers[k] = freq_brds[k-1] + (freq_brds[k] - freq_brds[k-1]) / 2;

    freq_brds = mel2freq(mel_brds);
    freq_centers = mel2freq(freq_centers);
        
    FILTERS = [];

    filter_descr = collections.namedtuple('filter', 'left center right')
    
    for k in range(n_bands):
  
        if(k == 0):
            L = freq_centers[k];
        else:
            L = freq_centers[k-1];
  
        C = freq_centers[k];

        if (k == n_bands - 1):
            R = freq_centers[k];
        else:
            R = freq_centers[k+1];
            
        tmp = filter_descr(left=L, right=R, center=C)
            
        FILTERS.append(tmp)

    return FILTERS

def get_freq_curves_for_filters(FILTERS, fs, fft_size, out_format = 0):

    reslen = len(FILTERS)
  
    FILTER_CURVES = []
    
    freq_step = float(fs) / float(fft_size);
    
    # OCTAVE version : freq_grid = 0 : freq_step : fs/2;
    freq_grid = numpy.arange(0.0, fs/2, freq_step)
    freq_grid = freq_grid[0 : -2]
    
    curve_descr = collections.namedtuple('curve_descr', 'x y')
    
    for k in range(reslen):
      
        x = numpy.array([FILTERS[k].left, FILTERS[k].center, FILTERS[k].right]);
        y = numpy.array([0.0, 1.0, 0.0]);
      
        st = numpy.ceil(x[0] / freq_step) * freq_step;
        # OCTAVE version : new_x = st : freq_step : x(3);
        new_x = numpy.arange(st, x[2], freq_step)
      
        if (out_format == 0): # frequency 
            # do nothing
            new_x = new_x
        elif (out_format == 1): # bin_number
            new_x = new_x / freq_step + 1;
            x = (x / freq_step + 1);
        else:
            raise Exception('unknown output format !!!');
        
        new_y = numpy.interp(new_x, x,y);
      
        tmp = curve_descr(x = (new_x - 1).astype('int32'), y = new_y)
      
        FILTER_CURVES.append(tmp)
      
    #endfor

    return FILTER_CURVES
#endfunction

def get_mel_fb_spectral_curve(spectrum, FILTERS, CURVES):

    n_filters = len(FILTERS)  
  
    freqs = numpy.zeros(n_filters)
    vals = numpy.zeros(n_filters)
  
    for k in range(n_filters):
        freqs[k] = FILTERS[k].center;
        idxs = CURVES[k].x;
        vals[k] = numpy.sum( spectrum[idxs] * CURVES[k].y) / numpy.sum(CURVES[k].y)
  
    return (freqs, vals)
#endfunction

def get_default_mel_fb_curve_elements(fs, mel_dim, spec_dim):
    fmin = 0.0
    fmax = fs / 2.0
    n_filters = mel_dim
    fft_size = (spec_dim - 1) * 2
    curve_type = 1

    FBANK = get_filter_bank_boarders(fmin, fmax, n_filters)
    FILTERS = get_freq_curves_for_filters(FBANK, fs, fft_size, curve_type)
    return (FBANK, FILTERS)

def transform_sp_data_to_mel_fb(data, fs, spec_dim, mel_dim, use_log = True, FBANK = None,
                                FILTERS = None):
    LOG_THRESHOLD = None

    new_data = numpy.zeros((data.shape[0], mel_dim), dtype='float32')
    
    if FBANK is None or FILTERS is None:
        (FBANK, FILTERS) = get_default_mel_fb_curve_elements(fs, mel_dim , spec_dim)   

    (mel_freq_bins, fb_curve) = get_mel_fb_spectral_curve(data, FILTERS = FBANK, CURVES = FILTERS)
    if use_log:
        fb_curve = transform_lin_to_log(fb_curve, LOG_THRESHOLD)
    new_data = fb_curve

    return new_data

def transform_mel_fb_data_to_sp(data, fs, mel_dim, spec_dim, use_log = True):
    LOG_THRESHOLD = None

    new_data = numpy.zeros((data.shape[0], spec_dim), dtype='float32')
    for idx in range(data.shape[0]):
        
        # perform actual interpolation in linear domain
        if use_log:
            lin_data = transform_log_to_lin(data[idx,:])
        else:
            lin_data = data[idx,:]
            #MY_DBG : observing this strange behaviod that when working in lin-domain, we can
            # actually generate negative envelopes which is definitely now allowed when talking
            # about spectral magnitude-envelopes
            lin_data[lin_data < 0.0] = 0.0000001
        sp_freq_bins, new_data[idx,:] = interp_spec_in_freq_uniform(lin_data, fs, spec_dim)
                     
    #endfor    
    return new_data


def get_spec_envelopes(sig_chunks):
    fft_size = sig_chunks.shape[1]
    num_chunks = sig_chunks.shape[0]
    assert(numpy.log2(fft_size) % 1.0) == 0 # make sure its a power of 2
    out_size = int(fft_size / 2 + 1)
    res = numpy.zeros( (num_chunks, out_size) )
    for k in range(num_chunks):
        res[k,:] = numpy.abs(numpy.fft.rfft(sig_chunks[k,:]))
    return res

def get_mel_fb_curves(env_chunks, fs, mel_dim):

    spec_dim = env_chunks.shape[1]
    num_chunks = env_chunks.shape[0]
    (FBANK, FILTERS) = get_default_mel_fb_curve_elements(fs, mel_dim, spec_dim)


    res = numpy.zeros( (num_chunks, mel_dim) )

    for k in range(num_chunks):
       res[k,:] = transform_sp_data_to_mel_fb(env_chunks[k,:], fs, spec_dim, mel_dim, use_log = False, 
            FBANK = FBANK, FILTERS = FILTERS)

    return res 

def get_spectral_flatness(spec):
    # make sure input is real-valued
    num_bins = len(spec)
    tmp = numpy.abs(spec)
    tmp[tmp == 0.0] = 0.000000001
    numer = numpy.exp( numpy.sum(numpy.log(tmp))/num_bins )
    denom = numpy.mean(tmp)
    res =  numer/ denom
    return res

def calc_spec_gram_flatness(sgram):
    # expect input where rows are spectrums in time and columns are freq-bins of each spectrum
    num_chunks = sgram.shape[0]
    res = numpy.zeros( (num_chunks, 1) )
    for k in range(num_chunks):
        tmp = get_spectral_flatness(sgram[k,:])
        res[k,0] = tmp
    return res
