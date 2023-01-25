import numpy as np
from sslf.sslf import Spectrum
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class zfinder():
    def __init__(self, transition, frequencies, flux, uncertainty, z_start=0, dz=0.01, z_end=10):
        self.transition = transition
        self.frequencies = frequencies
        self.flux = flux
        self.uncertainty = uncertainty
        self.z = np.arange(z_start, z_end+dz, dz)

    @staticmethod
    def _gaussf(x, a, s, x0):
        y = 0
        for i in range(1,11):
            y += (a * np.exp(-((x-i*x0) / s)**2)) # i = 1,2,3 ... 9, 10
        return y
    
    def _find_params(self):
        # Determine the best fitting parameters
        # try:
        params, covars = curve_fit(lambda x, a, s: self._gaussf(x, a, s, x0=self.loc), 
            self.frequencies, self.flux, bounds=[[0, (1/8)], [2*max(self.flux), (2/3)]], absolute_sigma=True) # best fit
        # except RuntimeError:
        #     return [], []
        return params, covars
    
    def _calc_reduced_chi2(self):

        # Calculate degrees of freedom
        flux_points = len(self.flux)
        num_gauss = 2*self.num_peaks
        DoF = flux_points - num_gauss - 1

        # calculate reduced chi2
        chi2 = sum(((self.flux - self.f_exp) / self.uncertainty)**2)
        reduced_chi2 = chi2 / DoF
        return reduced_chi2
    
    def _penalise_chi2(self):
        pass
    
    def _find_lines(self):
        # Create a line finder to find significant points
        s = Spectrum(self.flux)
        s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
        spec_peaks = s.channel_peaks
        spec_peaks = np.sort(spec_peaks) # sort the peaks from left to right instead of right to left
        return spec_peaks

    def zfind(self):

        # Initialise arrays
        self.all_chi2 = []
        
        # Interate through the list of redshifts and calculate the chi-squared
        for dz in self.z:

            # Calculate the offset of the gaussians
            self.loc = self.transition/(1+dz)

            # Calculate parameters of gaussian fit
            params, covars = self._find_params()

            # # If parameters don't exist, append the maximum chi2
            # if len(params) == 0:
            #     print(dz, 'damn')
            #     max_chi2 = max(self.all_chi2)
            #     self.all_chi2.append(max_chi2)
            #     continue
            
            # Calculate the expected flux array
            self.f_exp = self._gaussf(self.frequencies, a=params[0], s=params[1], x0=self.loc)

            # Find the number of gaussians overlayed
            peaks = find_peaks(self.f_exp)
            self.num_peaks = len(peaks[0])

            # Calculate the reduced chi-squared and check if it should be penalised
            reduced_chi2 = self._calc_reduced_chi2()
            # reduced_chi2 = self._penalise_chi2()

            # Parameter uncertainties
            perr = np.sqrt(np.diag(covars))

            self.all_chi2.append(reduced_chi2)
        
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
    freq, flux, uncert = gleam_0856.get_flux_axis()

    transition = 115.2712
    zf = zfinder(transition, freq, flux, uncert)
    z, chi2 = zf.zfind()
    plt.plot(z, chi2)
    plt.show()

if __name__ == '__main__':
    main()    