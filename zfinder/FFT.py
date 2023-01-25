# Import libraries
import csv
import numpy as np
import matplotlib.pyplot as plt

from Plot import count_decimals
from scipy.optimize import curve_fit
from uncertainty import Uncertainty

class FFT(object):

    def __init__(self, obj):
        ''' `FourierTransform` takes a `zFinder` object as input and computes the
            Fast Fourier Transform on a flux image. '''
        # # Only make animation for one plot otherwise taakes years to execute
        # if obj.num_plots > 1:
        #     raise ValueError('num_plots cannot be greater than 1')
        
        # Values
        self.z = obj.z
        self.peaks = obj.all_num_peaks
        self.files = obj.files
        self.sigma = obj.sigma
        self.d = count_decimals(obj.dz)
        self.nu = obj.nu
        self.ftransition = obj.ftransition
        x = obj.x_axis_flux
        y = obj.all_flux[0]

        N = 5*len(y) # Number of sample points
        T = x[1]-x[0] # sample spacing

        # Fourier transformed data
        self.yf = np.fft.rfft(y, N).real
        self.xf = np.fft.rfftfreq(N, T)

        # Store arrays
        self.fft_chi2 = []
        self.fft_parameters = []
        self.fft_uncertainty = []

        for dz, num_peak in zip(self.z, self.peaks[0]):
            try:
                params, covars = curve_fit(lambda x, a, c: self.double_damped_sinusoid(x, a, c, z=dz, 
                    nu=self.nu, f=self.ftransition), self.xf, self.yf, bounds=[[0.1, 0.1], [max(self.yf), 2]])
            except RuntimeError:
                self.fft_chi2.append(max(self.fft_chi2))
                continue
            
            # Calulate chi-squared
            y_obs = self.double_damped_sinusoid(self.xf, *params, z=dz, nu=self.nu, f=self.ftransition)

            chi2 = sum(((self.yf - y_obs)/(1))**2) # chi-squared
            reduced_chi2 = chi2 / (len(y_obs) - 2*num_peak - 1)

            perr = np.sqrt(np.diag(covars)) # parameter uncertainty

            self.fft_chi2.append(reduced_chi2)
            self.fft_parameters.append(params)
            self.fft_uncertainty.append(perr)

        self.index = np.argmin(self.fft_chi2) # index of the lowest redshift
        self.best_params = [round(i, 2) for i in self.fft_parameters[self.index]]
        self.y_fit = self.double_damped_sinusoid(self.xf, *self.best_params,
            z=self.z[self.index], nu=self.nu, f=self.ftransition)

        if self.files:
            self.export_csvs()

    @staticmethod
    def double_damped_sinusoid(x, a, c, z, nu, f):

        N = np.floor(((1+z)*nu/f)+1)

        p = 2*np.pi*(N*f/(1+z) - nu)
        q = 2*np.pi*((N+1)*f/(1+z) - nu)

        y = a*c*np.exp(-((x)**2) / (2*c**2)) * (np.cos(p*x) + np.cos(q*x))
        return y
    
    def plot_chi2_fft(self):
        plt.figure(figsize=(20,9))
        plt.plot(self.z, self.fft_chi2, color='black')
        plt.title(f'$\chi^2_r$ = {round(self.fft_chi2[self.index], 2)} @ z={round(self.z[self.index], self.d)}')
        plt.ylabel('$\chi^2_r$')
        plt.xlabel('Redshift')
        plt.yscale('log')
        plt.savefig('FFT Chi2', dpi=200)
        plt.show()
    
    def plot_fft(self):
        plt.figure(figsize=(20,9))
        plt.plot(self.xf, self.yf)
        plt.plot(self.xf, self.y_fit)
        plt.title(f'FFT Best Fit: z={round(self.z[self.index], self.d)}, a={self.best_params[0]}, c={self.best_params[1]}')
        plt.ylabel('Amplitude')
        plt.xlabel('Scale')
        plt.legend(['Flux FFT', 'Best Fit'])
        plt.savefig('FFT Best Fit')
        plt.show()

    def export_csvs(self):

        # Write best fitting parameters, redshift, and uncertainties.
        rows = zip(['amplitude', 'standard deviation'], self.fft_parameters[self.index], self.fft_uncertainty[self.index])
        with open('data.csv', 'a') as f:
            wr = csv.writer(f)
            wr.writerow(['Best Fitting Parameters for the FFT Plot'])
            wr.writerow(['Parameter', 'Value', 'Uncertainty'])
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

            num_peaks = self.peaks[0][self.index]
            reduced_sigma = self.sigma**2 / (len(self.yf) - 2*num_peaks - 1)
            uncert = Uncertainty(self.z, self.fft_chi2, reduced_sigma)
            neg, pos = uncert.calc_uncert()

            wr.writerow(['FFT Redshift and uncertainties'])
            wr.writerow(['-', 'z', '+'])
            wr.writerow([neg, self.z[np.argmin(self.fft_chi2)], pos])
            wr.writerow([])
        
        # Write chi2 vs redshift
        rows = zip(self.z, self.fft_chi2)
        with open('fft_chi2_data.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['Redshift', 'Chi-squared'])
            for row in rows:
                wr.writerow(row)
        
        # Write FFT Flux vs Scale + best fit
        rows = zip(self.xf, self.yf, self.y_fit)
        with open('fft_best_fit.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['Scales', 'y_fft', 'y_fit'])
            for row in rows:
                wr.writerow(row)