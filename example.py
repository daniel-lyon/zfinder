from RedshiftFinder import RedshiftFinder as zf

# Initial setup
filename = '0856_cube_c0.4_nat_80MHz_taper3.fits'
ra = [8, 56, 14.73] # Right Ascenion (h, m, s)
dec = [2, 23, 59.6] # Declination (d, m, s)
aperture_radius = 3 # Aperture Radius (pixels)
bvalue = 3 # BMAJ & BMIN (arcseconds)

# Spaced Points setup
min_sep = 5 # Minimum separation between points (pixels)
num_plots = 4 # Number of plots to make (#, must be a square number or 1)
circle_radius = 88 # Radius of the largest frequency (pixels)

# Redshift variables to iterate through
ftransition = 115.2712 # the first transition in GHz
z_start = 0 # initial redshift
dz = 0.01 # change in redshift
z_end = 10 # final redshift

zfind1 = zf(filename, ra, dec, aperture_radius, bvalue)
zfind1.circlePointVars(min_sep, num_plots, circle_radius)
zfind1.zfind(ftransition, z_start, dz, z_end, timer=True)

# Can use these arrays to create your own plots
all_chi2 = zfind1.allChi2
redshift_array = zfind1.z
plot_points = zfind1.coordinates
plot_colours = zfind1.plotColours
all_flux_y = zfind1.allFlux
all_flux_x = zfind1.xAxisFlux

# Otherwise, can use built in functions
zfind1.plotCoords()
zfind1.plotChi2()
zfind1.plotFlux()