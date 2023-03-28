import numpy as np
from flux_zfind import gaussf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def fft(x, y):
    """ Performs the fast fourier transform on the x and y axis """
    N = 5*len(y) # Number of sample points
    T = x[1]-x[0] # sample spacing

    # Fourier transformed data
    fflux = np.fft.rfft(y, N).real
    ffreq = np.fft.rfftfreq(N, T)
    return ffreq, fflux

def double_damped_sinusoid(x, a, s, z, nu, f):
    """ FFT Fitting function """
    A = 22.09797605*a*s
    C = 1/(np.pi*s)
    N = np.floor(((1+z)*nu/f)+1)
    p = 2*np.pi*(N*f/(1+z) - nu)
    q = 2*np.pi*((N+1)*f/(1+z) - nu)
    y = A*np.exp(-(x / C)**2) * (np.cos(p*x) + np.cos(q*x))
    return y

def find_params(transition, ffreq, fflux, dz, x0):
    """ Find the best fitting parameters for the FFT """
    try:
        params, covars = curve_fit(lambda x, a, s: double_damped_sinusoid(x, a, s, z=dz, 
            nu=x0, f=transition), ffreq, fflux, bounds=[[0, 0], [max(fflux), 2]])
    except:
        return np.array([None, None]), np.array([None, None])
    return params, covars

def calc_all_num_gauss(transition, frequency, dz):
    """ Calculate the number of gaussians inside the window (x-axis) """
    loc = transition/(1+dz)
    f_exp = gaussf(frequency, a=0.5, s=0.5, x0=loc)

    # Caclulate the number of gaussians overlayed
    gauss_peaks = find_peaks(f_exp)[0]
    return gauss_peaks

class fft_zfind(object):
    def __init__(self, transition, frequency, flux):
        """
        Find statistics of the source via fast fourier transform
        
        Parameters
        ----------
        transition : float
            The first transition frequency of the element or molecule to search for. Units
            follow frequency
        
        frequency : list
            A list of frequency values calculated from fits2flux
        
        flux : list
            A list of flux values calculated from fits2flux
        """
        
        self.transition = transition
        self.frequency = frequency
        self.flux = flux

    def fft_zfind(self, z_start=0, dz=0.01, z_end=10, sigma=1):
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
        
        sigma : float, optional
            The significance level of the uncertainty in the redshift 
            found at the minimum chi-squared. Default = 1
        
        Returns
        -------
        z : list
            The list of redshifts that was used to calculate the chi-squared
        
        chi2 : list
            A list of calculated chi-squared values
        """
        
        # Initialise lists
        z = np.arange(z_start, z_end+dz, dz)
        sigma = sigma
        all_chi2 = []
        fft_params = []
        fft_perrs = []
        all_num_peaks = []
        
        # Fourier transform frequency and flux
        ffreq, fflux = fft(self.frequency, self.flux)
        x0 = self.frequency[0]

        # Interate through the list of redshifts and calculate the chi-squared
        for dz in z:
            
            # Find the best fitting parameters at this redshift
            params, covars = find_params(self.transition, ffreq, fflux, dz, x0)
            if not params.tolist()[0]:
                all_chi2.append(max(all_chi2))
                fft_params.append([99,99])
                fft_perrs.append([99,99])
                all_num_peaks.append(0)
                continue
            perr = np.sqrt(np.diag(covars))

            # Calulate chi-squared
            fflux_obs = double_damped_sinusoid(ffreq, *params, z=dz, nu=x0, f=self.transition)

            # Find the number of gaussians that would be overlayed at this redshift
            gauss_peaks = calc_all_num_gauss(self.transition, self.frequency, dz)
            num_gauss_peaks = len(gauss_peaks)

            chi2 = sum((fflux - fflux_obs)**2) # chi-squared
            reduced_chi2 = chi2 / (len(fflux_obs) - 2*num_gauss_peaks - 1)

            # Append all items
            all_chi2.append(reduced_chi2)
            fft_params.append(params)
            fft_perrs.append(perr)
            all_num_peaks.append(num_gauss_peaks)
        
        lowest_index = np.argmin(all_chi2)
        self.params = fft_params[lowest_index]
        self.perr = fft_perrs[lowest_index]
        self.peaks = all_num_peaks[lowest_index]
        
        return z, all_chi2
    
    def fft_params(self):
        """
        Get the paramters of the best fitting redshift. Must run fft_zfind first.
        
        Returns
        -------
        params : list
            Best fitting redshift parameters -> [amplitude, standard deviation]
        
        perr : list
            Errors on the best fitting redshift parameters -> [amplitude, standard deviation] 
        """
        return self.params, self.perr