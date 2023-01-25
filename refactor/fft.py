import numpy as np

class fft():
    def __init__(self, frequency, flux):
        self.frequency = frequency
        self.flux = flux

    @staticmethod
    def _double_damped_sinusoid(x, a, c, z, nu, f):
        N = np.floor(((1+z)*nu/f)+1)
        p = 2*np.pi*(N*f/(1+z) - nu)
        q = 2*np.pi*((N+1)*f/(1+z) - nu)

        y = a*c*np.exp(-((x)**2) / (2*c**2)) * (np.cos(p*x) + np.cos(q*x))
        return y
    
    def _fft(self):
        N = 5*len(self.flux) # Number of sample points
        T = self.flux[1]-self.flux[0] # sample spacing

        # Fourier transformed data
        self.fflux = np.fft.rfft(self.flux, N).real
        self.ffreq = np.fft.rfftfreq(N, T)

    def fft_chi2(self):
        pass
