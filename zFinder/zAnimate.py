import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

class zAnimate(object):
    def __init__(self, obj):
        ''' `zAnimate` takes a `zFinder` object as input and computes animations
            of plots seen in `Plots
        '''
        # Only make animation for one plot otherwise takes years to execute
        if obj.num_plots > 1:
            raise ValueError('num_plots cannot be greater than 1')

        # Variables
        self.obj = obj
        self.x = self.obj.x_axis_flux
        self.y = self.obj.all_flux[0]
        self.c = self.obj.all_chi2[0]
        self.z = self.obj.z

    def animate_flux(self, savefile='flux_animation.gif'):
        ''' An animated plot of the flux at each frequency in a data cube being calculated. '''

        # Create the figure
        fig, axs = plt.subplots(figsize=(16,9))   

        # Animate by plotting from 0 -> 0,1 -> 0,1,2 -> etc
        @staticmethod
        def _flux_animation(i):
            axs.cla()
            axs.plot(self.x[0:i], self.y[0:i], color='black', drawstyle='steps-mid')
            axs.set_xlim(min(self.x), max(self.x))
            axs.set_ylim(min(self.y), max(self.y))
            axs.set_xlabel(f'Frequency $({self.obj.symbol}Hz)$')
            axs.set_ylabel('Flux $(mJy)$', x=0.01)
            axs.plot(self.x, np.zeros(len(self.x)), color='black', linestyle=(0, (5, 5)))
            axs.fill_between(self.x[0:i], self.y[0:i], 0, where=(self.y[0:i] > 0), color='gold', alpha=0.75)
        
        # Save animation
        ani = FuncAnimation(fig, _flux_animation, frames=len(self.x), interval=40, repeat=False)
        ani.save(savefile)    

    def animate_chi2(self, savefile='chi2_animation.gif'):   
        ''' An animated plot of how the chi-squared is calculated at every redshift. '''

        # Create the figure
        fig, axs = plt.subplots(figsize=(16,9))

        # Animate by plotting from 0 -> 0,1 -> 0,1,2 -> etc
        @staticmethod
        def _chi2_animation(i):
            axs.cla()
            axs.plot(self.z[0:i], self.c[0:i], color='black', drawstyle='steps-mid')
            axs.set_xlim(min(self.z), max(self.z))
            axs.set_ylim(min(self.c), max(self.c))
            axs.set_xlabel('Redshift')
            axs.set_ylabel('$\chi^2$', x=0.01)
            axs.set_title(f'min z={round(self.z[np.argmin(self.c[0:i+1])], 2)}')
        
        # Save animation
        ani = FuncAnimation(fig, _chi2_animation, frames=len(self.z), interval=20, repeat=False)
        ani.save(savefile) 
    
    def animate_redshift(self, savefile='redshift_animation.gif'):
        ''' An animation of how the redshift is iterated through. '''
        
        # Create the figure
        fig, axs = plt.subplots(figsize=(16,9))

        # Get the parameters
        params = self.obj.all_params[0]
        y0 = np.zeros(len(self.x))
        params = np.transpose(params)
        amp = params[0, :] # amplitude
        std = params[1, :] # standard deviation

        # Animate by plotting from 0 -> 0,1 -> 0,1,2 -> etc
        @staticmethod
        def _redshift_animation(i):
            axs.cla()
            axs.plot(self.x, y0, color='black', linestyle=(0, (5, 5)))
            axs.plot(self.x, self.y, color='black', drawstyle='steps-mid')
            axs.fill_between(self.x, self.y, 0, where=(self.y > 0), color='gold', alpha=0.75)
            axs.set_xlim(min(self.x), max(self.x))
            axs.set_ylim(min(self.y), max(self.y))
            axs.set_xlabel(f'Frequency $({self.obj.symbol}Hz)$')
            axs.set_ylabel('Flux $(mJy)$', x=0.01)
            axs.set_title(f'z={round(self.z[i], 2)}')
            loc = self.obj.ftransition/(1+self.z[i])
            f_exp = self.obj.gaussf(self.x, a=amp[i], s=std[i], x0=loc)
            axs.plot(self.x, f_exp, color='red')
        
        # Save animation
        ani = FuncAnimation(fig, _redshift_animation, frames=len(self.z), interval=10, repeat=False)
        ani.save(savefile)