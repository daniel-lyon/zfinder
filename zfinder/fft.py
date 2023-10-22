"""
Module for finding the redshift of a source using the fourier transform method.
"""

from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from zfinder.template import gaussf

def double_damped_sinusoid(x, a, s, z, nu, f):
    """ FFT Fitting function """
    A = 22.09797605*a*s
    C = 1/(np.pi*s)
    N = np.floor(((1+z)*nu/f)+1)
    p = 2*np.pi*(N*f/(1+z) - nu)
    q = 2*np.pi*((N+1)*f/(1+z) - nu)
    y = A*np.exp(-(x / C)**2) * (np.cos(p*x) + np.cos(q*x))
    return y

def calc_fft_params(transition, ffreq, fflux, dz, x0):
    """ Find the best fitting parameters for the FFT """
    params, covars = curve_fit(lambda x, a, s: double_damped_sinusoid(x, a, s, z=dz, 
        nu=x0, f=transition), ffreq, fflux, bounds=[[0, 0], [max(fflux), 2]])
    return params, covars

def fft(frequency, flux):
    """ 
    Performs the fast fourier transform on the frequency and flux axis 
    
    Parameters
    ----------
    frequency : list
        Frequency axis values found with fits2flux `get_freq()`
    
    flux : list
        Flux axis values found with fit2flux `get_flux()`
    
    Returns
    -------
    ffreq, fflux : list
        The Fourier Transformed frequency and flux axes respectively
    """
    N = 10*len(flux) # Number of sample points
    T = frequency[1]-frequency[0] # sample spacing

    # Fourier transformed data
    fflux = np.fft.rfft(flux, N).real
    ffreq = np.fft.rfftfreq(N, T)
    return ffreq, fflux

def _calc_all_num_gauss(transition, frequency, dz):
    """ Calculate the number of gaussians inside the window (x-axis) """
    loc = transition/(1+dz)
    f_exp = gaussf(frequency, a=0.5, s=0.5, x0=loc)

    # Caclulate the number of gaussians overlayed
    gauss_peaks = find_peaks(f_exp)[0]
    return gauss_peaks

def _process_fft_chi2_calculations(transition, frequency, uncertainty, ffreq, fflux, dz, x0):
    """ Use multiprocessing to significantly speed up chi2 calculations """
    # Find the best fitting parameters at this redshift. If fails return bad chi2
    try:
        params, _ = calc_fft_params(transition, ffreq, fflux, dz, x0)
    except RuntimeError:
        chi2 = sum((fflux)**2) * 1.2 # 1.2 is the penalising factor
        return chi2

    # Calulate chi-squared
    fflux_obs = double_damped_sinusoid(ffreq, a=params[0], s=params[1], z=dz, nu=x0, f=transition)

    # Find the number of gaussians that would be overlayed at this redshift
    gauss_peaks = _calc_all_num_gauss(transition, frequency, dz)

    chi2 = sum(((fflux - fflux_obs) / uncertainty)**2) # chi-squared
    reduced_chi2 = chi2 / (len(fflux_obs) - 2*len(gauss_peaks) - 1)
    return reduced_chi2

def _mp_fft_zfind(transition, frequency, uncertainty, ffreq, fflux, z, verbose=True):
    """ Process the FFT chi2 calculations in parallel """
    with Pool() as pool:
        jobs = [pool.apply_async(_process_fft_chi2_calculations, (transition, frequency, uncertainty, ffreq, fflux, dz, frequency[0])) for dz in z]
        chi2 = [res.get() for res in tqdm(jobs, disable=not verbose)]
    return chi2

def _serial_fft_zfind(transition, frequency, uncertainty, ffreq, fflux, z, verbose=True):
    """ Process the FFT chi2 calculations serially """
    chi2 = [_process_fft_chi2_calculations(transition, frequency, uncertainty, ffreq, fflux, dz, frequency[0]) for dz in tqdm(z, disable=not verbose)]
    return chi2

def fft_zfind(transition, frequency, flux, uncertainty=1, z_start=0, dz=0.01, z_end=10, verbose=True, parallel=True):
    """ 
    Finds the best redshift by performing the fast fourier transform on the flux data. The
    chi-squared is caclulated at every redshift by iterating through delta-z. The most 
    likely redshift corresponds to the minimum chi-squared.
    
    Parameters
    ----------        
    z_start : int, optional
        The first value in the redshift list. Default = 0
        
    dz : float, optional
        The change in redshift. Default = 0.01
    
    z_end : int, optional
        The final value in the redshift list. Default = 10
    
    verbose : bool, optional
        If True, print progress. Default = False
    
    Returns
    -------
    z : list
        The list of redshifts that was used to calculate the chi-squared
    
    chi2 : list
        A list of calculated chi-squared values
    """
    
    # Initialise lists
    z = np.arange(z_start, z_end+dz, dz)
      
    # Fourier transform frequency and flux
    ffreq, fflux = fft(frequency, flux)

    # Parallelise the chi2 calculations
    if verbose:
        print('Calculating FFT fit chi-squared values...')
    if parallel:
        chi2 = _mp_fft_zfind(transition, frequency, uncertainty, ffreq, fflux, z, verbose=verbose)
    else:
        chi2 = _serial_fft_zfind(transition, frequency, uncertainty, ffreq, fflux, z, verbose=verbose)
    return z, chi2

def fft_per_pixel(transition, frequency, all_flux, z_start=0, dz=0.01, z_end=10, size=3):
    """ Perform the FFT fitting method on all flux values """
    print('Calculating all FFT fit chi-squared values...')
    verbose, parallel, uncertainty = False, False, 1
    with Pool() as pool:
        jobs = [pool.apply_async(fft_zfind, (transition, frequency, flux, uncertainty, z_start, dz, z_end, verbose, parallel)) for flux in all_flux]
        results = [res.get() for res in tqdm(jobs)]
    all_z = [z[np.argmin(chi2)] for z, chi2 in results]
    z = np.reshape(all_z, (size, size))
    return z