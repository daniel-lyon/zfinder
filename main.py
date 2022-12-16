from RedshiftFinder import RedshiftFinder, zf_plotter

# Variable settings
aperture_radius = 3 # Aperture Radius (pixels)
bvalue = 3 # BMAJ & BMIN (arcseconds)
min_sep = 1 # Minimum separation between points (pixels)
num_plots = 1 # Number of plots to make (1, 5, 10, 15, 20, 25, integer multiples of 25)
circle_radius = 85 # Radius of the largest frequency (pixels)
ftransition = 115.2712 # the first transition in GHz
z_start = 0 # initial redshift
dz = 0.01 # change in redshift
z_end = 11 # final redshift

'''##########--- 0856 ---##########'''
image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
ra = [8, 56, 14.8] # Right Ascenion (h, m, s) orig_ra = [8, 56, 14.73]
dec = [2, 24, 0.6, 1] # Declination (d, m, s, sign)  orig_dec = [2, 23, 59.6, 1]

'''##########--- 0913 ---##########''' # big dif in ap rad
# image = '0913_cube_c0.4_nat_80MHz_taper3.fits'
# ra = [9, 13, 37.14] # Right Ascenion (h, m, s) orig_ra = [9, 13, 37.14]
# dec = [2, 31, 45.4, 1] # Declination (d, m, s, sign) orig_dec = [2, 31, 45.4, 1]

'''##########--- 0917 ---##########''' # big dif in ap rad
# image = '0917_cube_c0.4_nat_80MHz_taper3.fits' 
# ra = [9, 17, 34.36] # Right Ascenion (h, m, s) orig_ra = [9, 17, 34.36]
# dec = [0, 12, 42.7, -1] # Declination (d, m, s, sign) orig_dec = [0, 12, 42.7, -1]

'''##########--- 0918 ---##########''' # small dif in ap rad
# image = '0918_cube_c0.4_nat_80MHz_taper3.fits'
# ra = [9, 18, 23.2] # Right Ascenion (h, m, s) orig_ra = [9, 18, 23.2]
# dec = [0, 5, 5, -1] # Declination (d, m, s, sign) orig_dec = [0, 5, 5, -1]

# Plotting
zfind1 = RedshiftFinder(image, ra, dec, aperture_radius, bvalue)
zfind1.circle_point_vars(min_sep, num_plots, circle_radius)
z = zfind1.zfind(ftransition, z_start, dz, z_end)

zf1 = zf_plotter(zfind1)
zf1.plot_points()
zf1.plot_flux()
zf1.plot_chi2()
mean, std = zf1.plot_hist_chi2()
zf1.plot_snr_scale()