import numpy as np

def gen_transfer_matrix(x, angle, wl):
    return np.exp(-2j*np.pi * x[:,None]* np.sin(angle[None,:])/wl)


def plane_waves(sensor_location, source_location, angle, wavelength, gain_sig):
    # plane wave propagation (sensors,sources)
    tm = gen_transfer_matrix(sensor_location, angle, wavelength)
    ##If loop to simulate random gain-phase perturbation in non-fading channel
    if gain_sig!=0:
        #af = 1: non-fading channel ; af  = 0: Rayleigh fading
        af = 1
        gain_matrix = np.random.normal(loc=af, scale=gain_sig, 
                                          size=(sensor_location.shape[0],angle.shape[0], 2)).view(np.complex128)
        gain_matrix = gain_matrix.reshape(sensor_location.shape[0],angle.shape[0])
        tm = np.multiply(tm, gain_matrix)
         
    signal = np.einsum("ij, jl -> il", tm, source_location)
    return signal # single wavelength


def scm(sig): 
    return np.matmul(sig, np.conj(sig.T))/sig.shape[1]


def cbf(A,signal): 
    return np.real(np.diag(np.conj(A.T)@ scm(signal) @ A))


def sbl_map(A,X):
    """
    Sparse Bayesian Learning beamformer
    parameters: 
        A sensing matrix 
            (Nsensors, Ndirections) or (Nsensors, Ndirections, N_freqavg)
        X signal matrix
            (N_sensors, N__snapshots) or (N_sensors, N__snapshots, N_freqavg)
        directions wavenumber vector k_xyz
    output:
       P_SBL SBL map
    """
    from sbl import Options, SBL
    options = Options(
            convergence_error         = 1e-3,
            gamma_range               = 1e-4,
            convergence_maxiter       = 400,
            Nsource                   = 3,
            convergence_min_iteration = 1,
            status_report             = 1,
            fixedpoint                = 1,
            flag                      = 0, # iteration flag
            )
    return SBL(np.atleast_3d(A),np.atleast_3d(X),options)[0]


all_beamformers = {'sbl': sbl_map, 'cbf' : cbf}


def extract_peaks(signals, n_peaks):
    """ find the most relevant peaks in 1D signals. prepends zeros if fewer
    peaks than requested are found.
    parameters:
        signals (n_signals, len_signal) beamforming signals
        n_peaks int                     number of peaks to search for
    returns: 
        peaks   (n_snapshots, n_peaks)  indices of peaks in signals
    """
    from scipy.signal import find_peaks
    pdata = np.zeros((signals.shape[0],n_peaks),dtype=int)
    for ii, signal in enumerate(signals):
        # find peaks
        peaks = find_peaks(signal)[0]
        # if only fewer peaks can be found, prepend zeros
        lp    = np.min([len(peaks),n_peaks])
        # select lp peaks with the most power, sorted from low to high
        pdata[ii][-lp:] = peaks[np.argsort(signal[peaks])][-lp:]
        # sort with index
        pdata[ii] = np.sort(pdata[ii])
    return pdata
