# Required functions
from astropy.io import fits
from functions import wcs2pix, fits_flux, gaussf, arrayfix
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import curve_fit

# Ignore the background warnings (there are hundreds)
warnings.filterwarnings("ignore", module='photutils.background')
warnings.filterwarnings("ignore", module='astropy.wcs.wcs')
warnings.filterwarnings("ignore", module='scipy.optimize')

# Read .fits image
# filename = '0856_cube_c0.4_nat_80MHz_taper2.fits' 
filename = '0856_cube_c0.4_nat_80MHz_taper3.fits'    
image = fits.open(filename)

# Get the header
hdr = image[0].header
data = image[0].data[0]

# Specified values
RA = [8, 56, 14.73] # RA to find at
DEC = [2, 23, 59.6] # DEC to find at
radius = 3 # Radius of the aperture
bvalue = 3 # arcseconds

# Convert world coordinates to pix
x, y = wcs2pix(RA, DEC, hdr)

# Get fluxes and ucnertainties at each image
y_flux, uncert = fits_flux(image, (x, y), radius, bvalue)
y_flux *= 1000
uncert *= 1000

# Convert x-axis to Hz
freq_start = hdr['CRVAL3']/10**9 # GHz
freq_incr = hdr['CDELT3']/10**9 # GHz
freq_len = np.shape(data)[0] # length
freq_end = freq_start + freq_len * freq_incr # where to stop
x = np.linspace(freq_start, freq_end, freq_len) # axis to plot

# Redshift parameter setup
z_start = 0
dz = 0.01
z_end = 10
z_n = int((1/dz)*(z_end-z_start))+1 # total number of redshifts to iterate through
z = np.linspace(z_start, z_end, z_n)

# Gaussian Model parameter setup
chi2_vs_z = []
amplitude = 1
y0 = 0

# For every redshift, calculate the corresponding chi squared value
for ddz in z:
    mean = 115.2712/(1+ddz)

    popt, pcov = curve_fit(lambda x, a: gaussf(x, a, mean, y0), x, y_flux, absolute_sigma=True)
    
    f_exp = gaussf(x, *popt, mean, y0=y0)

    uncert = arrayfix(uncert) # average 0's from values left & right

    chi2 = sum(((y_flux - f_exp) / uncert)**2)

    chi2_vs_z.append(chi2)

# Chi^2 vs Redshift plot
plt.plot(z, chi2_vs_z)
plt.xlabel('Redshift ($Z$)')
plt.ylabel('$\chi^2$')
# plt.yscale('log')
plt.text(5.25, 1, 'CO: z=5.55', color='red')
plt.savefig('fig2.png', dpi=200)
plt.show()

# Flux vs Frequency Plot
plt.figure(figsize=(15,5))
plt.plot([84.2, 115], [0, 0], color='black', linestyle='--', dashes=(5,5))
plt.plot(x, y_flux, color='black', drawstyle='steps-mid')
plt.plot(x, gaussf(x, *popt, mean, y0=y0), color='red')
# plt.title('z=')
plt.fill_between(x, y_flux, 0, where=(y_flux > 0), color='gold', alpha=0.5)
plt.xlabel('Frequency $(GHz)$')
plt.ylabel('Flux $(mJy)$')
plt.margins(x=0)
plt.savefig('fig1.png', dpi=200)
plt.show()