import csv
import warnings

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import Angle

from zfinder.fits2flux import Fits2flux, wcs2pix
from zfinder.fft import fft_zfind

warnings.filterwarnings("ignore", module='astropy.wcs.wcs')

# TODO: add template_per_pixel function
# TODO: merge get_all_flux to Fits2flux class --> update get_flux to take in a list of ra and dec
# TODO: Automatically get the minimum and maximum velocities

def radec2str(ra, dec):
    """ Convert RA and DEC to a string """
    ra = Angle(ra, unit=u.degree).to_string(unit=u.hour, sep=':', precision=4)
    dec = Angle(dec, unit=u.degree).to_string(unit=u.degree, sep=':', precision=4)
    return ra, dec

def generate_square_pix_coords(size, target_x, target_y):
    """ Generate a list of coordinates for a square of pixel coordinates around a target pixels """
    matrix = np.arange(size) - size//2
    x, y = np.meshgrid(matrix + target_x, matrix + target_y)
    x, y = x.ravel(), y.ravel()
    return x, y

def generate_square_world_coords(fitsfile, ra, dec, size):
    """ Generate a list of coordinates for a square of ra & dec around a target ra & dec """
    # Generate x, y pix coordinates around target ra & dec
    hdr = fits.getheader(fitsfile)
    target_pix_ra_dec = wcs2pix(ra, dec, hdr)
    x, y = generate_square_pix_coords(size, *target_pix_ra_dec)

    # Convert x, y pix coordinates to world ra and dec
    wcs = WCS(hdr, naxis=2)
    ra, dec = wcs.all_pix2world(x, y, 1)
    ra, dec = radec2str(ra, dec)
    return ra, dec

def get_all_flux(fitsfile, ra, dec, aperture_radius):
    """ Get the flux values for all ra and dec coordinates """
    print('Calculating all flux values...')
    all_flux = []
    for r, d in tqdm(zip(ra, dec), total=len(ra)):
        flux, flux_uncert = Fits2flux(fitsfile, r, d, aperture_radius).get_flux(verbose=False)
        all_flux.append(flux)
    return all_flux

def fft_per_pixel(transition, frequency, all_flux, z_start=0, dz=0.01, z_end=10, size=3):
    """ Doc string here """

    # Calculate the chi-squared values
    print('Calculating FFT fit chi-squared values...')
    all_z = []
    for flux in tqdm(all_flux):
        z, chi2 = fft_zfind(transition, frequency, flux, z_start, dz, z_end, verbose=False)
        all_z.append(z[np.argmin(chi2)])

    # Reshape the array
    z = np.reshape(all_z, (size, size))
    return z      

def _write_csv_rows(filename, data):
    """ Doc string here """
    with open(filename, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(data)

def main():
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    size = 5
    transition = 115.2712
    z_start = 5.5
    dz = 0.001
    z_end = 5.6
    aperture_radius = 0.5

    
    # z = fft_per_pixel(transition, frequency, all_flux, z_start, dz, z_end, size)
    # z = np.array([[5.55, 5.549, 5.548], [5.549, 5.548, 5.548], [5.549, 5.548, 5.547]])
    z = np.array([[5.551,5.550,5.549,5.548,5.547],
                [5.551,5.550,5.549,5.548,5.547],
                [5.550,5.549,5.548,5.548,5.547],
                [5.549,5.549,5.548,5.547,5.547],
                [5.549,5.548,5.548,5.547,5.547]])
    _write_csv_rows('fft_per_pixel.csv', z)

    # Calculate the velocities
    target_z = np.take(z, z.size // 2) # redshift of the target ra and dec
    velocities = 3*10**5*((((1 + target_z)**2 - 1) / ((1 + target_z)**2 + 1)) - (((1 + z)**2 - 1) / ((1 + z)**2 + 1))) # km/s
     
    # Generate the x and y coordinates to iterate through
    hdr = fits.getheader(fitsfile)
    target_pix_ra_dec = wcs2pix(ra, dec, hdr)
    x, y = generate_square_pix_coords(size, *target_pix_ra_dec)

    # velocities = np.flipud(velocities)
    import matplotlib.pyplot as plt
    w = WCS(hdr, naxis=2)
    plt.subplot(projection=w)
    hm = plt.imshow(velocities, cmap='bwr', interpolation='nearest', vmin=-50, vmax=50,
            extent=[x[0], x[-1], y[0], y[-1]],  
            origin='lower')
    plt.colorbar(hm, label='km/s')
    plt.xlabel('RA', fontsize=15)      
    plt.ylabel('DEC', fontsize=15)      
    plt.savefig('fft_per_pixel.png')
    plt.show()

if __name__ == '__main__':
    main()