from zfinder import zfinder

# Variable settings
transition = 115.2712 # the first transition in GHz
aperture_radius = 3 # Aperture Radius (pixels)
bvalue = 3 # BMAJ & BMIN (arcseconds)
z_start = 4.28 # initial redshift
dz = 0.0001 # change in redshift
z_end = 4.31 # final redshift

'''##########--- 0856 ---##########'''
# image = '0856_cube_c0.4_nat_80MHz_taper3.fits'
# ra = [8, 56, 14.8] # Right Ascenion (h, m, s) original ra = [8, 56, 14.73]
# dec = [2, 24, 0.6, 1] # Declination (d, m, s, sign)  original dec = [2, 23, 59.6, 1]

'''##########--- 0913 ---##########'''
# image = '0913_cube_c0.4_nat_80MHz_taper3.fits'
# ra = [9, 13, 37.16] # Right Ascenion (h, m, s) original ra = [9, 13, 37.14]
# dec = [2, 31, 45.6, 1] # Declination (d, m, s, sign) original dec = [2, 31, 45.4, 1]

'''##########--- 0917 ---##########'''
# image = '0917_cube_c0.4_nat_80MHz_taper3.fits' 
# ra = [9, 17, 34.36] # Right Ascenion (h, m, s) original ra = [9, 17, 34.36]
# dec = [0, 12, 42.7, -1] # Declination (d, m, s, sign) original dec = [0, 12, 42.7, -1]

'''##########--- 0918 ---##########'''
# image = '0918_cube_c0.4_nat_80MHz_taper3.fits'
# ra = [9, 18, 23.2] # Right Ascenion (h, m, s) original ra = [9, 18, 23.2]
# dec = [0, 5, 5, -1] # Declination (d, m, s, sign) original dec = [0, 5, 5, -1]





""" SPT_0243-49 """
# image = 'SPT_0243-49.contsub_clean.image.fits'
# image = 'SPT_0243-49.contsub.clean.taper.image.fits'
# image = 'SPT_0243-49.contsub.clean.taper5.image.fits'
# ra = [2, 43, 8.81]
# dec = [-49, 15, 35, -1]

""" SPT_0300-46 """
# image = 'SPT_0300-46.contsub.clean.image.fits'
# ra = [3, 0, 4.37]
# dec = [-46, 21, 24.3, -1]

""" SPT_0319-47 """
# image = 'SPT_0319-47.contsub.clean.taper.image.fits'
# ra = [3, 19, 31.88]
# dec = [-47, 24, 33.7, -1]

""" SPT_0345-47 """
image = 'SPT_0345-47.contsub.clean.taper.image.fits'
ra = [3, 45, 10.77]
dec = [-47, 25, 39.5, -1]

""" SPT_0346-52 """
# image = 'SPT_0346-52.contsub.clean.taper.image.fits'
# ra = [3, 46, 41.33]
# dec = [-52, 5, 2.1, -1]





""" ALMA J0216 """
# image = 'OBJ2.image.fits'
# ra = [2, 16, 18.61]
# dec = [-33, 1, 52.2, -1]

""" ALMA J0201 """
# image = 'OBJ3.image.fits'
# ra = [2, 1, 18.43]
# dec = [-34, 41, 3.5, -1]

""" ALMA J0240 """
# image = 'OBJ4.image.fits'
# ra = [2, 40, 20.18]
# dec = [-32, 7, 0.9, -1]






# Find the best fit redshift
source = zfinder(image, ra, dec, transition, aperture_radius, bvalue)
# source.zflux(z_start=z_start, dz=dz, z_end=z_end)
# source.zfft(z_start=z_start, dz=dz, z_end=z_end, reduction=True)
# source.random_stats(n=101, radius=50, spread=1)
source.fft_per_pixel(size=51, z_start=4.28,  dz=0.0001, z_end=4.31)