"""
Module for finding the redshift of a source using the gaussian template shifting method.
"""

import warnings
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from sslf.sslf import Spectrum
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated", category=OptimizeWarning)

def gaussf(x, a, s, x0):
    """ Function to fit a sum of gaussians """
    y = 0
    for i in range(1,30):
        y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 29,30
    return y

def _calc_reduced_chi2(flux, flux_expected, flux_uncertainty, gauss_lines):
    """ Calculate the reduced chi-squared: chi2_r = chi2 / (num_points - free_params - 1) """
    gauss_dofs = 2*len(gauss_lines) # 2 DoF per gaussian (amp, std)
    DoF = len(flux) - gauss_dofs - 1 # Calculate the degrees of freedom

    # Calculate chi-squared
    chi2 = sum(((flux - flux_expected) / flux_uncertainty)**2)

    # Calculate reduced chi-squared
    reduced_chi2 = chi2 / DoF
    return reduced_chi2

def _penalise_chi2(gauss_lines, sslf_lines):
    """ Penalise chi-squared values that do not fit to lines found with sslf line finder """
    penalising_factor = 1.2 # factor multiplying bad fits
    non_penalising_factor = 1 # factor multiplying good fits
    tolerance = 3 # minimum number of channels separating gauss and sslf lines
    
    # Sort Ascending
    gauss_lines = np.sort(gauss_lines)
    sslf_lines = np.sort(sslf_lines)

    # If the number of lines are not equal, return penalising factor
    if len(sslf_lines) != len(gauss_lines):
        return penalising_factor

    # If the lines are not within a range of each other, return penalising factor
    for bline, gline in zip(sslf_lines, gauss_lines):
        if gline not in range(bline-tolerance, bline+tolerance):
            return penalising_factor
        
    # If lines are within range, return non penalising factor
    else:
        return non_penalising_factor

def find_lines(flux):
    """ Create a line finder to find significant points """
    s = Spectrum(flux)
    s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
    spec_peaks = s.channel_peaks

    # Calculate the ratio of the snrs and scales
    snrs = [round(i,2) for i in s.peak_snrs] # the snrs of the peaks
    scales = [i[1]-i[0] for i in s.channel_edges] # the scales of the peaks

    # Sort lists
    sorted_lists = zip(*sorted(zip(spec_peaks, snrs, scales)))
    try:
        spec_peaks, snrs, scales = map(list, sorted_lists)
    except ValueError:
        spec_peaks, snrs, scales = [], [], [] # when there are 0 lines found
    return spec_peaks, snrs, scales

def calc_template_params(frequency, flux, observed_transition):
    params, covars = curve_fit(lambda x, a, s: gaussf(x, a, s, x0=observed_transition), 
        frequency, flux, bounds=[[0, (1/8)], [2*max(flux), (2/3)]], absolute_sigma=True)
    return params, covars    
    
def _process_template_chi2_calculations(transition, frequency, flux, flux_uncertainty, sslf_lines, dz):
    """ Use multiprocessing to significantly speed up chi2 calculations """
    # Calculate the frequency of the first observed transition line
    observed_transition = transition / (1 + dz)

    # Fit a gaussian template to the flux
    try:
        params, _ = calc_template_params(frequency, flux, observed_transition) # best fit
    except RuntimeError:
        chi2 = sum((flux)**2) * 1.2
        return chi2
    
    # Calculate the flux of a perfect function fit
    flux_expected = gaussf(frequency, a=params[0], s=params[1], x0=observed_transition)

    # Finds the list of Gaussians lines present at this redshift and inside the window (in channel no.#)
    gauss_lines = find_peaks(flux_expected)[0] # E.g. = [60, 270]

    # Calculate reduced chi2
    multiplier = _penalise_chi2(gauss_lines, sslf_lines)
    reduced_chi2 = _calc_reduced_chi2(flux, flux_expected, flux_uncertainty, gauss_lines) * multiplier
    return reduced_chi2

def _mp_template_zfind(transition, frequency, flux, flux_uncertainty, z, sslf_lines, verbose=True):
    """ Process the Template chi2 calculations in parallel """
    with Pool() as pool:
        jobs = [pool.apply_async(_process_template_chi2_calculations, 
            (transition, frequency, flux, flux_uncertainty, sslf_lines, dz)) for dz in z]
        chi2 = [result.get() for result in tqdm(jobs, disable=not verbose)]
    return chi2

def _serial_template_zfind(transition, frequency, flux, flux_uncertainty, z, sslf_lines, verbose=True):
    """ Process the Template chi2 calculations serially """
    chi2 = [_process_template_chi2_calculations(transition, frequency, flux, flux_uncertainty, sslf_lines, dz) for dz in tqdm(z, disable=not verbose)]
    return chi2

def template_zfind(transition, frequency, flux, flux_uncertainty=1, z_start=0, dz=0.01, z_end=10, verbose=True, parallel=True):
    """
    Using the gaussian template shifting method, calculate the reduced chi-squared at every change
    in redshift.

    Parameters
    ----------
    z_start : float, optional
        The redshift to start calculated chi2 fits at. Default=0

    dz : float, optional
        Change in redshift to iterature through. Default=0.01

    z_end : float, optional
        Final redshift to start calculating chi2 at. Default=10

    verbose : bool, optional
        If True, print progress. Default=True

    Returns
    -------
    z : list
        List of redshift values that were iterated through

    chi2 : list
        List of calculated chi-squared values.

    Example
    -------
    ```python
    z, chi2 = template_zfind()

    lowest_index = np.argmin(chi2)
    best_fit_redshift = z[lowest_index]
    ```
    """
    # Create a list of redshifts to iterate through
    z = np.arange(z_start, z_end+dz, dz)
    sslf_lines, _, _ = find_lines(flux) # E.g. = [60, 270]

    # Parallelise slow loop to execute much faster
    if verbose:
        print('Calculating Template fit chi-squared values...')
    if parallel:
        chi2 = _mp_template_zfind(transition, frequency, flux, flux_uncertainty, z, sslf_lines, verbose=verbose)
    else:
        chi2 = _serial_template_zfind(transition, frequency, flux, flux_uncertainty, z, sslf_lines, verbose=verbose)
    return z, chi2

def template_per_pixel(transition, frequency, all_flux, all_flux_uncertainty, z_start=0, dz=0.01, z_end=10, size=3):
    """ Perform the Template fitting method on all flux values """
    print('Calculating all Template fit chi-squared values...')
    verbose, parallel = False, False
    with Pool() as pool:
        jobs = [pool.apply_async(template_zfind, (transition, frequency, flux, flux_uncert, z_start, dz, z_end, verbose, parallel)) 
            for flux, flux_uncert in zip(all_flux, all_flux_uncertainty)]
        results = [res.get() for res in tqdm(jobs)]
    all_z = [z[np.argmin(chi2)] for z, chi2 in results]
    z = np.reshape(all_z, (size, size))
    return z