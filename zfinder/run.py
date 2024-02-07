def main():
    from zfinder import zfinder

    # fitsfile = 'zfinder/SPT_0345-47.contsub.clean.taper.image.original.fits'
    # ra = '03:45:10.77'
    # dec = '-47:25:39.5'
    fitsfile = 'zfinder/0856_cube_c0.4_nat_80MHz_taper3.fits'
    ra = '08:56:14.8'
    dec = '02:24:00.6'
    
    aperture_radius = 3
    transition = 115.271

    source = zfinder(fitsfile, ra, dec, aperture_radius, transition, savefig=True, export=False)
    # source.flux_channels(channels=[[35,65], [250,280]])
    # source.template(z_start=5, dz=0.001, z_end=6)
    source.export_line_map(channels=[[35,65], [250,280]])
    
    # source.export_line_map(channels=[[46, 55], [263, 271]])
    
    # z_start = 5
    # dz = 0.001
    # z_end = 6
    # source.template(z_start, dz, z_end)
    

  

if __name__ == '__main__':
    main()