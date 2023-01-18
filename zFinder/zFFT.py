# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

class zFFT(object):

    def __init__(self, obj):
        ''' `FourierTransform` takes a `zFinder` object as input and computes the
            Fast Fourier Transform on a flux image. '''

        # # Only make animation for one plot otherwise taakes years to execute
        # if obj.num_plots > 1:
        #     raise ValueError('num_plots cannot be greater than 1')
        
        # Values
        self.z = obj.z
        nu = obj.nu
        ftransition = obj.ftransition
        x = obj.x_axis_flux
        y = obj.all_flux[0]

        N = 5*len(y) # Number of sample points
        T = x[1]-x[0] # sample spacing

        # Fourier transformed data
        yf = np.fft.rfft(y, N).real
        xf = np.fft.rfftfreq(N, T)

        # Store chi-squareds
        self.fft_chi2 = []

        for dz in self.z:
            try:
                params, covars = curve_fit(lambda x, a, c: self.double_damped_sinusoid(x, a, c, z=dz, nu=nu, f=ftransition), 
                    xf, yf, bounds=[[1, 0.1], [10, 2]])
            except RuntimeError:
                self.fft_chi2.append(max(self.fft_chi2))
                continue
            
            # Calulate chi-squared
            y_obs = self.double_damped_sinusoid(xf, *params, z=dz, nu=nu, f=ftransition)

            chi2 = sum(((yf - y_obs)/(1))**2)

            self.fft_chi2.append(chi2)
    
    @staticmethod
    def double_damped_sinusoid(x, a, c, z, nu, f):

        N = np.floor(((1+z)*nu/f)+1)

        p = 2*np.pi*(N*f/(1+z) - nu)
        q = 2*np.pi*((N+1)*f/(1+z) - nu)

        y = a*c*np.exp(-((x)**2) / (2*c**2)) * (np.cos(p*x) + np.cos(q*x))
        return y
    
    def plot_fft(self):
        index = np.argmin(self.fft_chi2) # index of the lowest redshift
        plt.figure(figsize=(20,9))
        plt.plot(self.z, self.fft_chi2, color='black')
        plt.title(f'$\chi^2$ = {round(self.fft_chi2[index], 2)} @ z={round(self.z[index], 2)}')
        plt.ylabel('$\chi^2$')
        plt.xlabel('Redshift')
        plt.savefig('FFT', dpi=200)
        plt.show()