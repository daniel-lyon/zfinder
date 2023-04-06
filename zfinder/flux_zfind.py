import numpy as np
from sslf.sslf import Spectrum
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def find_lines(flux):
    """ Create a line finder to find significant points """
    s = Spectrum(flux)
    s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
    spec_peaks = s.channel_peaks

    # Calculate the ratio of the snrs and scales
    snrs = [round(i,2) for i in s.peak_snrs] # the snrs of the peaks
    scales = [i[1]-i[0] for i in s.channel_edges] # the scales of the peaks
    return spec_peaks, snrs, scales

def gaussf(x, a, s, x0):
    """ Function to fit a sum of gaussians """
    y = 0
    for i in range(1,30):
        y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
    return y

def _find_params(frequency, flux, loc):
    """ Find the best fitting parameters for the gaussians """
    params, covars = curve_fit(lambda x, a, s: gaussf(x, a, s, x0=loc), 
        frequency, flux, bounds=[[0, (1/8)], [2*max(flux), (2/3)]], absolute_sigma=True) # best fit
    return params, covars

class Template():
    def __init__(self, transition, frequency, flux, uncertainty=1):
        """
        Find statistics of the source via gaussian overlays
        
        Parameters
        ----------
        transition : float
            The first transition frequency of the element or molecule to search for. Units
            follow frequency
        
        frequency : list
            A list of frequency values calculated from fits2flux
        
        flux : list
            A list of flux values calculated from fits2flux

        uncertainty : list, optional
            A list of uncertainty values calculated from fits2flux. If left unset, resulting 
            chi-squared statistics will be relative. Default = 1
        """
        self._transition = transition
        self._frequency = frequency
        self._flux = flux
        self._uncertainty = uncertainty
    
    @staticmethod
    def __calc_reduced_chi2(flux, f_exp, gauss_peaks, uncertainty):
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
    def __penalise_chi2(gauss_peaks, blind_lines):
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
    
    def zfind(self, z_start=0, dz=0.01, z_end=10, sigma=1, penalise=True):
        """ 
        Finds the best redshift by fitting gaussian functions overlayed on flux data. The
        chi-squared is calculated at every redshift by iterating through delta-z. The most 
        likely redshift corresponds to the minimum chi-squared.
        
        Parameters
        ----------
        z_start : int, optional
            The first value in the redshift list. Default = 0
            
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value in the redshift list. Default = 10
        
        sigma : float, optional
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared. Default = 1
            
        penalise : bool, optional
            If True, perform chi-squared penalisation with sslf. Default = True
        
        Returns
        -------
        z : list
            The list of redshifts that was used to calculate the chi-squared
        
        chi2 : list
            A list of calculated chi-squared values
        """
        
        # Initialise
        z = np.arange(z_start, z_end+dz, dz)
        sigma = sigma
        all_chi2 = []
        all_params = []
        all_num_peaks = []
        all_perrs = []
        self._blind_lines, self._snrs, self._scales = find_lines(self._flux)
        
        # Interate through the list of redshifts and calculate the chi-squared
        for dz in z:

            # Calculate the offset of the gaussians
            loc = self._transition/(1+dz)

            # Calculate parameters of gaussian fitS
            params, covars = _find_params(self._frequency, self._flux, loc)
            perr = np.sqrt(np.diag(covars)) # calculate the error on the gaussian parameters
            
            # Calculate the expected flux array (gaussian overlay)
            f_exp = gaussf(self._frequency, a=params[0], s=params[1], x0=loc)

            # calculate the number of gaussians overlayed
            gauss_peaks = find_peaks(f_exp)[0]

            # Calculate the reduced chi-squared and check if it should be penalised
            if penalise:
                multiplier = self.__penalise_chi2(gauss_peaks, self._blind_lines)
            else:
                multiplier = 1
            reduced_chi2 = self.__calc_reduced_chi2(self._flux, f_exp, gauss_peaks, self._uncertainty) * multiplier

            all_chi2.append(reduced_chi2)
            all_params.append(params)
            all_num_peaks.append(len(gauss_peaks))
            all_perrs.append(perr)
        
        lowest_index = np.argmin(all_chi2)
        self._params = all_params[lowest_index]
        self._perr = all_perrs[lowest_index]
        self._peaks = all_num_peaks[lowest_index]
        return z, all_chi2
    
    def parameters(self):
        """
        Get the paramters of the best fitting redshift. Must run gauss_zfind first.
        
        Returns
        -------
        params : list
            Best fitting redshift parameters -> [amplitude, standard deviation]
        
        perr : list
            Errors on the best fitting redshift parameters -> [amplitude, standard deviation] 
        """
        return self._params, self._perr
    
    def sslf(self):
        """ 
        Get the statistics of the lines found with sslf. Must run gauss_zfind first.
        
        Returns
        -------
        peaks : list
            The channel locations of each significant points.
        
        snrs : list
            The signal-to-noise ratio of each significant points.
        
        scales : list
            The scale of each significant point
        """
        return self._blind_lines, self._snrs, self._scales 