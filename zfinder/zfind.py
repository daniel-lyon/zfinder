# Import packages
import numpy as np

from sslf.sslf import Spectrum
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class zfind():
    """
    Doc string
    """
    def __init__(self, transition, frequency, flux, uncertainty=1):
        self._transition = transition
        self._frequency = frequency
        self._flux = flux
        self._uncertainty = uncertainty

    @staticmethod
    def gaussf(x, a, s, x0):
        """ Function to fit a sum of gaussians """
        y = 0
        for i in range(1,30):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 29,30
        return y
    
    @staticmethod
    def _calc_reduced_chi2(flux, f_exp, gauss_peaks, uncertainty):
        """ Calculate the reduced chi-squared: chi2_r = chi2 / (num_points - free_params - 1) """

        # Calculate the degrees of freedom
        flux_points = len(flux)
        num_gauss = len(gauss_peaks)
        gauss_dofs = 2*num_gauss # 2 DoF per gaussian (amp, std)
        DoF = flux_points - gauss_dofs - 1

        # Calculate chi-squared
        chi2 = sum(((flux - f_exp) / uncertainty)**2)

        # Calculate reduced chi-squared
        reduced_chi2 = chi2 / DoF
        return reduced_chi2
    
    @staticmethod
    def _penalise_chi2(gauss_peaks, blind_lines):
        """ Penalise chi-squared values that do not fit to lines found with sslf line finder """

        # Variable Settings
        gauss_peaks = np.sort(gauss_peaks)
        blind_lines = np.sort(blind_lines)
        
        num_gaussians = len(gauss_peaks)
        num_blind_lines = len(blind_lines)
        penalising_factor = 1.2
        tolerance = 3

        # If the number of lines are not equal, return penalising factor
        if num_blind_lines != num_gaussians:
            return penalising_factor

        # If the lines are not within a range of each other, return penalising factor
        for bline, gline in zip(blind_lines, gauss_peaks):
            if gline not in range(bline-tolerance, bline+tolerance):
                return penalising_factor
        else:
            return 1 # No penalisation found, return factor of 1
    
    @staticmethod
    def _find_lines(flux):
        """ Create a line finder to find significant points """
        s = Spectrum(flux)
        s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
        spec_peaks = s.channel_peaks
        return spec_peaks
    
    @staticmethod
    def _find_params(frequency, flux, loc):
        """ Find the best fitting parameters for the gaussians """
        params, covars = curve_fit(lambda x, a, s: zfind.gaussf(x, a, s, x0=loc), 
            frequency, flux, bounds=[[0, (1/8)], [2*max(flux), (2/3)]], absolute_sigma=True) # best fit
        return params, covars

    def zfind(self, z_start=0, dz=0.01, z_end=10):

        chi2 = []

        z = np.arange(z_start, z_end+dz, dz)
        sslf_lines = self._find_lines(self._flux)

        for dz in z:
            
            observed_transition = self._transition / (1 + dz)

            params, covars = self._find_params(self._frequency, self._flux, observed_transition)
            
            flux_expected = zfind.gaussf(self._frequency, a=params[0], s=params[1], x0=observed_transition)

            gauss_peaks = find_peaks(flux_expected)[0]

            multiplier = self._penalise_chi2(gauss_peaks, sslf_lines)
            reduced_chi2 = self._calc_reduced_chi2(self._flux, flux_expected, gauss_peaks, self._uncertainty) * multiplier
        
            chi2.append(reduced_chi2)

        return z, chi2