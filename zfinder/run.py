from finder import zfinder

def main():
    """ GLEAM J0856+0200 """
    # fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    # ra = '08:56:14.8' # target
    # dec = '02:24:00.6'
    
    # # ra = '08:56:14.76' # target
    # # dec = '02:23:59.676'
    
    # # ra = '08:56:14.9' # small field
    # # dec = '02:24:05.1'

    # aperture_radius = 3
    # transition = 115.2712
    # size = 15
    # z_start = 5.5
    # dz = 0.001
    # z_end = 5.6
    # aperture_radius_pp = 0.5

    # source = zfinder(fitsfile, ra, dec, aperture_radius, transition)
    # # source.template(z_start, dz, z_end, parallel=True)
    # # source.fft(z_start, dz, z_end, parallel=True)
    # source.fft_pp(size, z_start, dz, z_end, aperture_radius_pp)
    # source.template_pp(size, z_start, dz, z_end, aperture_radius_pp)





    """ SPT 0345-47 """
    image = 'zfinder/SPT_0345-47.contsub.clean.taper.image.original.fits'
    ra = '03:45:10.77'
    dec = '-47:25:39.5'

    aperture_radius = 3
    transition = 115.2712
    size = 3
    z_start = 4.28
    dz = 0.0001
    z_end = 4.31
    aperture_radius_pp = 0.5
    
    source = zfinder(image, ra, dec, aperture_radius, transition)
    # source.template(z_start, dz, z_end)
    # source.fft(z_start, dz, z_end)
    source.fft_pp(size, z_start, dz, z_end, aperture_radius_pp)
    source.template_pp(size, z_start, dz, z_end, aperture_radius_pp)

if __name__ == '__main__':
    main()