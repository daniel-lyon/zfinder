# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from decimal import Decimal
from sslf.sslf import Spectrum
from scipy.stats import binned_statistic

def count_decimals(number: float):
    ''' count the amount of numbers after the decimal place '''
    d = Decimal(str(number))
    d = abs(d.as_tuple().exponent)
    return d

def flatten_list(input_list: list[list]):
    ''' Turns lists of lists into a single list '''
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

class zPlot(object):
    def __init__(self, obj, plots_per_page=25):

        # TODO: Add number of rows (or maybe columns instead?) to automatically calculate plots per page
        # TODO: Change saving to work for multiple pages

        ''' `Plots` takes a `zFinder` object as an input to easily compute plots
        used for statistical analysis.
        '''

        # Values
        self.obj = obj

        # The number of pages of data to plot
        self.pages = self.obj.num_plots // plots_per_page
        if self.pages == 0:
            self.pages = 1
        
        # The number of rows and columns used for each page
        if self.obj.num_plots >= 5:
            self.cols = 5
            self.rows = self.obj.num_plots // (self.cols * self.pages)
            self.squeeze = True
        else:
            self.cols = 1
            self.rows = 1
            self.squeeze = False
        
    @staticmethod
    def plot_peaks(y_axis, x_axis, plot_type):

        # TODO: Find a way to remove this function? (Combine with other Spectrum function?)
        # TODO: Fix text plotting of snrs and scales
        # TODO: Animations for flux's and chi-squared's

        ''' Plot the found peaks of a line finder on top of another plot.

        Parameters
        ----------
        y_axis : `list`
            The y-axis with which to find the significant peaks

        x_axis : `list`
            The x-axis with which to plot the significant peaks

        plot_type : 'matplotlib.axes._subplots.AxesSubplot'
            : The figure to plot the peaks on 
        '''

        s = Spectrum(y_axis)
        s.find_cwt_peaks(scales=np.arange(4,10), snr=3)
        peaks = s.channel_peaks

        scales = [i[1]-i[0] for i in s.channel_edges]
        snrs = [round(i,2) for i in s.peak_snrs]
        snr_text = [0.35 if i%2==0 else -0.35 for i in range(len(s.peak_snrs))]
        scale_text = [0.40 if i%2==0 else -0.40 for i in range(len(scales))]

        for i, snr, s_text, sc_text, sc in zip(peaks, snrs, snr_text, scale_text, scales):
            plot_type.plot(x_axis[i], y_axis[i], marker='o', color='blue')
            plot_type.text(x_axis[i], s_text, s=f'snr={snr}', color='blue')
            plot_type.text(x_axis[i], sc_text, s=f'scale={sc}', color='blue')
    
    # @staticmethod
    # def show():
    #     plt.show()
    # 
    # @staticmethod
    # def savefig():
    #     plt.savefig()

    def plot_points(self):
        ''' Plot the distribution of coordinates. '''

        circle_points = np.transpose(self.obj.coordinates)
        points_x = circle_points[0, :] # all x coordinates except the first which is the original
        points_y = circle_points[1, :] # all y coordinates except the first which is the original
        circ = plt.Circle((self.obj.centre_x, self.obj.centre_y), self.obj.circle_radius, fill=False, color='blue')
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(7)
        ax.add_patch(circ)
        plt.title('Distribution of spaced random points')
        plt.scatter(points_x, points_y, color=self.obj.plot_colours)
        plt.xlim(-self.obj.circle_radius-1+self.obj.centre_x, self.obj.circle_radius+1+self.obj.centre_x)
        plt.ylim(-self.obj.circle_radius-1+self.obj.centre_y, self.obj.circle_radius+1+self.obj.centre_y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show() # Show the plot
    
    def plot_chi2(self):
        ''' Plot the chi-squared vs redshift at every coordinate. '''

        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        AllColours = np.array_split(self.obj.plot_colours, self.pages)
        d = count_decimals(self.obj.dz)
        
        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for page, (chi2, colours) in enumerate(zip(all_chi2, AllColours)):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, figsize=(20,9), 
                tight_layout=True, sharex=True, sharey=True, squeeze=self.squeeze)
            fig.supxlabel('Redshift')
            fig.supylabel('$\chi^2$', x=0.01)
            axs = axs.flatten()

            # Plot the chi-squared(s) and redshift
            for index, (c2, color) in enumerate(zip(chi2, colours)):
                lowest_redshift = self.obj.z[np.argmin(c2)]
                axs[index].plot(self.obj.z, c2, color=color)
                axs[index].plot(lowest_redshift, min(c2), 'bo', markersize=5)
                axs[index].set_title(f'$\chi^2$ = {round(min(c2), 2)} @ z={round(lowest_redshift, d)}')
                axs[index].set_yscale('log')

            if self.obj.num_plots == 1:
                plt.savefig('Chi-squared Plot', dpi=300)
            else:
                plt.savefig(f'Chi-squared Plot ({page}).png', dpi=300)
            plt.show() # Show the plot

    def plot_flux(self):

        # TODO: Change outside boarder colour to use all_colours

        ''' Plot the flux vs frequency at every coordinate. '''

        # Split data into pages
        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        all_flux = np.array_split(self.obj.all_flux, self.pages)
        all_params = np.array_split(self.obj.all_params, self.pages)
        d = count_decimals(self.obj.dz) # decimal places to round to
        x_axis = self.obj.x_axis_flux

        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for page, (fluxes, chi2, params) in enumerate(zip(all_flux, all_chi2, all_params)):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, figsize=(20,9), 
                tight_layout=True, sharex=True, sharey=True, squeeze=self.squeeze)
            fig.supxlabel(f'Frequency $({self.obj.symbol}Hz)$')
            fig.supylabel('Flux $(mJy)$')
            axs = axs.flatten()

            # Plot the flux(s) and best fit gaussians
            for index, (flux, c2, param) in enumerate(zip(fluxes, chi2, params)):
                lowest_index = np.argmin(c2)
                lowest_redshift = self.obj.z[lowest_index]
                axs[index].plot(x_axis, np.zeros(len(x_axis)), color='black', linestyle=(0, (5, 5))) # dashed line at y=0 from left to right
                axs[index].plot(x_axis, flux, color='black', drawstyle='steps-mid') # 'histogram' type scatter of the flux at every frequency
                axs[index].plot(x_axis, self.obj.gaussf(x_axis, *param[lowest_index], 
                    x0=self.obj.ftransition/(1+lowest_redshift)), color='red') # gaussian overlay at best fit redshift
                axs[index].margins(x=0)
                axs[index].fill_between(x_axis, flux, 0, where=(flux > 0), color='gold', alpha=0.75)
                axs[index].set_title(f'z={round(lowest_redshift, d)}')
                self.plot_peaks(flux, self.obj.x_axis_flux, axs[index])

            if self.obj.num_plots == 1:
                plt.savefig('Flux Best Fit', dpi=300)
            else:
                plt.savefig(f'Flux Best fit ({page}).png', dpi=300)
            plt.show()
    
    def plot_hist_chi2(self):
        ''' Plot a histogram of the chi-squared at every coordinate.
        
        Returns
        -------
        bin_mean : `list`
            The mean value of each bin

        bin_std : `list`
            The standard deviation of each bin
        '''

        # Initialise return arrays
        bin_std = []
        bin_mean = []

        # Split data into pages
        all_chi2 = np.array_split(self.obj.all_chi2, self.pages)
        colours = np.array_split(self.obj.plot_colours, self.pages)
        
        # Plot the reduced chi-squared histogram(s) across multiple pages (if more than one)
        for page, color in zip(all_chi2, colours):

            # Setup the figure and axes
            fig, axs = plt.subplots(self.rows, self.cols, tight_layout=True, sharey=True, squeeze=self.squeeze)
            fig.supxlabel('$\chi^2$') 
            fig.supylabel('Count (#)')
            axs = axs.flatten()

            for index, (chi2, c) in enumerate(zip(page, color)):
                reduced_chi2 = np.array(chi2)/(len(chi2)-1)
                axs[index].hist(reduced_chi2, 20, color=c)
                axs[index].set_yscale('log')

                # Calculate the mean and standard deviation of each bin
                mean = binned_statistic(reduced_chi2, reduced_chi2, 'mean', 20)
                std = binned_statistic(reduced_chi2, reduced_chi2, 'std', 20)

                # Flip the arrays to be read left to right and append to the return arrays
                bin_mean.append(np.flip(mean[0]))
                bin_std.append(np.flip(std[0]))
            plt.show()

        return bin_mean, bin_std

    def plot_snr_scales(self):
        ''' Plot a histogram of the snr and scale. '''

        snrs = flatten_list(self.obj.all_snrs)
        scales = flatten_list(self.obj.all_scales)

        # Setup the figure and axes
        fig, (ax_snr, ax_scale) = plt.subplots(1, 2, sharey=True)
        fig.supylabel('Count (#)')

        # Plot the snrs histogram(s)
        ax_snr.hist(snrs, 20)
        ax_snr.set_title('SNR histogram')
        ax_snr.set_xlabel('SNR')

        # Plot the scales histogram
        ax_scale.hist(scales, [8,10,12,14,16,18,20])
        ax_scale.set_title('Scales Histogram')
        ax_scale.set_xlabel('Scale')
        plt.show()