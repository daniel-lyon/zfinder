# Redshift Finding Algorithm

<h1 align="left">
  <img src="https://github.com/daniel-lyon/zfinder/blob/main/Affiliations/zfinder-logo.png">
</h1>

<h1 align="center">
  <img src="https://github.com/daniel-lyon/ICRAR-Monster-Black-Holes/blob/main/Affiliations/icrar_logo.png" width="111">
  <img src="https://github.com/daniel-lyon/ICRAR-Monster-Black-Holes/blob/main/Affiliations/redshift.png" width="550">
  <img src="https://github.com/daniel-lyon/ICRAR-Monster-Black-Holes/blob/main/Affiliations/qut_logo.jpg" width="111">
</h1>

ICRAR summer studentship project "Monster Black Holes at The Dawn of Time" (https://www.icrar.org/study-with-icrar/studentships/2022-studentship-projects/monster-black-holes-at-the-dawn-of-time/). Using the transition lines of emission spectra, the most likely redshift is determined by fitting gaussian lines to flux data. 

Example Usage Source: J085614 + 022400
----------

<h1 align="left">
  <img src="https://github.com/daniel-lyon/ICRAR-Monster-Black-Holes/blob/main/Figures/0856_flux.png">
  <img src="https://github.com/daniel-lyon/ICRAR-Monster-Black-Holes/blob/main/Figures/0856_chi2.png">
</h1>

Methodology
----------

A three dimensional `.fits` data cube of right ascension (ra), declination (dec), and frequency is required to find the redshift. Given a target ra and dec, the flux at the target location is repeatedly calculated for all frequencies. To find the redshift of the source, the chi-squared is calculated at small changes in redshift. The minimum chi-squared corresponds to the most likely redshift of the source.

<h1 align="center">
  <img src="https://github.com/daniel-lyon/zfinder/blob/main/Animations/flux_animation.gif" width="800">
  <img src="https://github.com/daniel-lyon/zfinder/blob/main/Animations/redshift_animation.gif" width="800">
  <img src="https://github.com/daniel-lyon/zfinder/blob/main/Animations/chi2_animation.gif" width="800">
</h1>
