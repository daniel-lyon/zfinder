import numpy as np
from zflux import zflux
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def fft(x, y):
    N = 5*len(y) # Number of sample points
    T = x[1]-x[0] # sample spacing

    # Fourier transformed data
    fflux = np.fft.rfft(y, N).real
    ffreq = np.fft.rfftfreq(N, T)
    return ffreq, fflux

class zfft():
    def __init__(self, transition, frequency, flux):
        """
        Finds the redshift of an FFT'd source by fitting a double damped sinusoid
        to the Fourier transformed data. This is done at small changes in redshift,
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
        self.transition = transition
        self.frequency = frequency
        self.flux = flux

    @staticmethod
    def _double_damped_sinusoid(x, a, s, z, nu, f):
        """ FFT Fitting function """
        O = f/(1+z)
        A = 22.09797605*a*s
        C = 1/(np.pi*s)
        N = np.floor(nu/O+1)
        p = 2*np.pi*(N*O - nu)
        q = 2*np.pi*((N+1)*O - nu)
        y = A*np.exp(-(x / C)**2) * (np.cos(p*x) + np.cos(q*x))
        return y

    def _find_params(self):
        """ Find the best fitting parameters for the FFT """
        try:
            params, covars = curve_fit(lambda x, a, s: self._double_damped_sinusoid(x, a, s, z=self.dz, 
                nu=self.x0, f=self.transition), self.ffreq, self.fflux, bounds=[[0, 0], [max(self.fflux), 2]])
        except:
            return np.array([None, None]), np.array([None, None])
        return params, covars
    
    def _calc_all_num_gauss(self):
        """ Calculate the number of gaussians inside the window (x-axis) """
        loc = self.transition/(1+self.dz)
        f_exp = zflux._gaussf(self.frequency, a=0.5, s=0.5, x0=loc)

        # Caclulate the number of gaussians overlayed
        gauss_peaks = find_peaks(f_exp)[0]
        return gauss_peaks

    def zfind(self, z_start=0, dz=0.01, z_end=10, sigma=1):
        """
        Iterate through small changes in redshift and caclulate the chi-squared at each dz 
        
        Parameters
        ----------
        z_start : int, optional
            The beginning of the redshift list. Default = 0
        
        dz : float, optional
            The change in redshift. Default = 0.01
        
        z_end : int, optional
            The final value of the redshift list. Defualt = 10
        
        sigma : float
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared
        
        Returns
        -------
        z : list
            The list of redshifts that was used to calculate the chi-squared
        
        chi2 : list
            A list of calculated chi-squared values
        """

        # Initialise lists
        self.z = np.arange(z_start, z_end+dz, dz)
        self.sigma = sigma
        self.all_chi2 = []
        self.fft_params = []
        self.fft_perrs = []
        self.all_num_peaks = []
        
        # Fourier transform frequency and flux
        self.ffreq, self.fflux = fft(self.frequency, self.flux)
        self.x0 = self.frequency[0]

        # Interate through the list of redshifts and calculate the chi-squared
        for dz in self.z:
            self.dz = dz
            
            # Find the best fitting parameters at this redshift
            params, covars = self._find_params()
            if not params.tolist()[0]:
                self.all_chi2.append(max(self.all_chi2))
                self.fft_params.append([99,99])
                self.fft_perrs.append([99,99])
                self.all_num_peaks.append(0)
                continue
            perr = np.sqrt(np.diag(covars))

            # Calulate chi-squared
            fflux_obs = self._double_damped_sinusoid(self.ffreq, *params, z=dz, nu=self.x0, f=self.transition)

            gauss_peaks = self._calc_all_num_gauss()
            num_gauss_peaks = len(gauss_peaks)

            chi2 = sum(((self.fflux - fflux_obs)/(1))**2) # chi-squared
            reduced_chi2 = chi2 / (len(fflux_obs) - 2*num_gauss_peaks - 1)

            self.all_chi2.append(reduced_chi2)
            self.fft_params.append(params)
            self.fft_perrs.append(perr)
            self.all_num_peaks.append(num_gauss_peaks)
        
        return self.z, self.all_chi2

def main():
    from fits2flux import fits2flux
    import matplotlib.pyplot as plt

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
    z, chi2 = test.zfind()
    ffreq, fflux = fft(freq, flux)

    plt.plot(ffreq, fflux)
    plt.show()

    plt.plot(z, chi2)
    plt.show()

if __name__ == '__main__':
    main()