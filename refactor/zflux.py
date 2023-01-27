import numpy as np
from sslf.sslf import Spectrum
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def find_lines(flux):
    """ Create a line finder to find significant points """
    s = Spectrum(flux)
    s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
    spec_peaks = s.channel_peaks
    spec_peaks = np.sort(spec_peaks) # sort the peaks ascending   
    num_spec_peaks = len(spec_peaks)
    return spec_peaks, num_spec_peaks

class zflux():
    def __init__(self, transition, frequency, flux, uncertainty, z_start=0, dz=0.01, z_end=10):
        """ 
        Finds the best redshift by fitting gaussian functions overlayed on flux data. The
        chi-squared is caclulated at every redshift by iterating through dz. The most likely
        redshift of the source corresponds to the minimum chi-squared.

        Parameters
        ----------
        transition : float
            The first transition frequency of the element or molecule to search for. Units
            follow frequency
        
        frequency : list
            A list of frequency values calculated from fits2flux
        
        flux : list
            A list of flux values caclulated from fits2flux

        uncertainty : list
            A list of uncertainty values caclulated from fits2flux
        """
        self.transition = transition
        self.frequency = frequency
        self.flux = flux
        self.uncertainty = uncertainty
        self.z = np.arange(z_start, z_end+dz, dz)

    @staticmethod
    def _gaussf(x, a, s, x0):
        """ Function to fit a sum of gaussians in a window (x-axis)"""
        y = 0
        for i in range(1,11):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def _find_params(self):
        """ Find the best fitting parameters for the gaussians """
        params, covars = curve_fit(lambda x, a, s: self._gaussf(x, a, s, x0=self.loc), 
            self.frequency, self.flux, bounds=[[0, (1/8)], [2*max(self.flux), (2/3)]], absolute_sigma=True) # best fit
        return params, covars
    
    def _calc_reduced_chi2(self):
        """ Caclulate the reduced chi-squared: chi2_r = chi2 / (num_points - free_params - 1) """

        # Caclulate the degrees of freedom
        flux_points = len(self.flux)
        len_peaks = len(self.gauss_peaks)
        num_gauss = 2*len_peaks # 2 DoF per gaussian (amp, std)
        DoF = flux_points - num_gauss - 1

        # Calculate chi-squared
        chi2 = sum(((self.flux - self.f_exp) / self.uncertainty)**2)

        # Calculate reduced chi-squared
        reduced_chi2 = chi2 / DoF
        return reduced_chi2
    
    def _penalise_chi2(self):
        """ Penalise chi-squared values that do not fit to lines found with sslf line finder"""

        # Variable Settings (Make **kwargs?)
        num_gaussians = len(self.gauss_peaks)
        penalising_factor = 1.2
        tolerance = 3

        # If the number of lines are not equal, return penalising factor
        if self.num_blind_lines != num_gaussians:
            return penalising_factor

        # If the lines are not within a range of each other, return penalising factor
        for bline, gline in zip(self.blind_lines, self.gauss_peaks):
            if gline not in range(bline-tolerance, bline+tolerance):
                return penalising_factor
        else:
            return 1 # No penalisation found, return factor of 1

    def zfind(self):
        """ Iterate through small changes in redshift and caclulate the chi-squared at each dz """

        # Initialise arrays
        self.all_chi2 = []
        self.all_num_peaks = []
        self.blind_lines, self.num_blind_lines = find_lines(self.flux)
        
        # Interate through the list of redshifts and calculate the chi-squared
        for dz in self.z:

            # Calculate the offset of the gaussians
            self.loc = self.transition/(1+dz)

            # Calculate parameters of gaussian fit
            params, covars = self._find_params()
            
            # Calculate the expected flux array
            self.f_exp = self._gaussf(self.frequency, a=params[0], s=params[1], x0=self.loc)

            # Find the number of gaussians overlayed
            self.gauss_peaks = find_peaks(self.f_exp)[0]

            # Calculate the reduced chi-squared and check if it should be penalised
            multiplier = self._penalise_chi2()
            reduced_chi2 = self._calc_reduced_chi2() * multiplier

            self.all_chi2.append(reduced_chi2)
            self.all_num_peaks.append(len(self.gauss_peaks))
        
        return self.z, self.all_chi2

def main():
    import matplotlib.pyplot as plt
    from fits2flux import fits2flux

    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    aperture_radius = 3
    bvalue = 3
    gleam_0856 = fits2flux(image, ra, dec, aperture_radius, bvalue)
    freq = gleam_0856.get_freq()
    flux, uncert = gleam_0856.get_flux()

    transition = 115.2712
    zf = zflux(transition, freq, flux, uncert, dz=0.01)
    z, chi2 = zf.zfind()
    plt.plot(z, chi2)
    plt.show()

if __name__ == '__main__':
    main()    