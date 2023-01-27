import numpy as np
from zflux import zflux, find_lines
from uncertainty import uncertainty
from scipy.optimize import curve_fit

def fft(x, y):
    N = 5*len(y) # Number of sample points
    T = x[1]-x[0] # sample spacing

    # Fourier transformed data
    fflux = np.fft.rfft(y, N).real
    ffreq = np.fft.rfftfreq(N, T)
    return ffreq, fflux

class zfft(zflux):
    def __init__(self, transition, frequency, flux, z_start=0, dz=0.01, z_end=10):
        """
        Finds the redshift of an FFT'd source by fitting a double damped sinusoid
        to the Fourier transformed data. This is done at small changes in redshift
        and for each dz, the chi-squared is calculated. The minimum chi-squared
        corresponds to the most likely redshift.

        Parameters
        ----------
        transition : float
            The first transition frequency of the element or molecule to search for. Units
            follow frequency

        frequency : list
            A list of frequency values calculated from fits2flux

        flux : list
            A list of flux values caclulated from fits2flux
        """
        super().__init__(transition, frequency, flux, z_start, dz, z_end)
        self.z = np.arange(z_start, z_end+dz, dz)

    @staticmethod
    def _double_damped_sinusoid(x, a, c, z, nu, f):
        N = np.floor(((1+z)*nu/f)+1)
        p = 2*np.pi*(N*f/(1+z) - nu)
        q = 2*np.pi*((N+1)*f/(1+z) - nu)

        y = a*c*np.exp(-((x)**2) / (2*c**2)) * (np.cos(p*x) + np.cos(q*x))
        return y

    def _find_params(self):
        params, covars = curve_fit(lambda x, a, c: self._double_damped_sinusoid(x, a, c, z=self.dz, 
            nu=self.frequency[0], f=self.transition), self.ffreq, self.fflux, bounds=[[0.1, 0.1], [max(self.fflux), 2]])
        return params, covars
    
    def _param_uncert(self):
        lowest_index = np.argmin(self.fft_chi2)
        perr = self.fft_perr[lowest_index]
        return perr
    
    def _z_uncert(self):
        lowest_index = np.argmin(self.fft_chi2)
        # num_peaks = self.all_num_peaks[lowest_index]
        num_peaks = 2 # assume always 2 gaussians
        reduced_sigma = self.sigma**2 / (len(self.flux) - 2*num_peaks - 1)
        uncert = uncertainty(self.z, self.fft_chi2, reduced_sigma)
        neg, pos = uncert.calc_uncert()
        return neg, pos

    def zfind(self, sigma=1):

        # Initialise return arrays
        self.fft_chi2 = []
        self.fft_params = []
        self.fft_perr = []
        self.sigma = sigma
        
        # Fourier transform frequency and flux
        self.ffreq, self.fflux = fft(self.frequency, self.flux)

        # FInd the number of lines in the regular flux data
        spec_peaks, num_spec_peaks = find_lines(self.flux)

        # Interate through the list of redshifts and calculate the chi-squared
        for dz in self.z:
            self.dz = dz
            
            # Find the best fitting parameters at this redshift
            params, covars = self._find_params()
            perr = np.sqrt(np.diag(covars))

            # Calulate chi-squared
            fflux_obs = self._double_damped_sinusoid(self.ffreq, *params, z=dz, nu=self.frequency[0], f=self.transition)

            chi2 = sum(((self.fflux - fflux_obs)/(1))**2) # chi-squared
            reduced_chi2 = chi2 / (len(fflux_obs) - 2*num_spec_peaks - 1)

            self.fft_chi2.append(reduced_chi2)
            self.fft_params.append(params)
            self.fft_perr.append(perr)

        neg, pos = self._z_uncert()
        perr = self._param_uncert()
        
        return self.z, self.fft_chi2, [neg, pos], perr

def main():
    from fits2flux import fits2flux
    import matplotlib.pyplot as plt
    transition = 115.2712
    image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = [8, 56, 14.8]
    dec = [2, 24, 0.6, 1]
    aperture_radius = 3
    bvalue = 3
    gleam_0856 = fits2flux(image, ra, dec, aperture_radius, bvalue)
    freq = gleam_0856.get_freq()
    flux, uncert = gleam_0856.get_flux()
    transition = 115.2712

    test = zfft(transition, freq, flux)
    z, chi2, z_uncert, perr = test.zfind()
    ffreq, fflux = fft(freq, flux)

    print(z_uncert)
    print(perr)

    plt.plot(ffreq, fflux)
    plt.show()

    plt.plot(z, chi2)
    plt.show()

if __name__ == '__main__':
    main()