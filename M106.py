import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.nddata import NDData, NDDataRef, StdDevUncertainty, CCDData, Cutout2D
from astropy.coordinates import SkyCoord
import astropy.visualization as viz
from scipy.ndimage import gaussian_filter as gauss


### This code takes a FITS file and produces images from the CCD data
### FITS data is courtesy of the Hubble Legacy Archive at https://hla.stsci.edu/hlaview.html

hdulist = fits.open('M106.fits') #fetch the file

#getting necessary header values
photflam = hdulist[0].header['PHOTFLAM'] # inverse sensitivity of telescope
exptime = hdulist[0].header['EXPTIME']  # total exposure time


#Get data in correct units
img = CCDData.read('M106.fits', unit='adu')  # get data from the file
img2 = img.copy() #copy data to work with
img2.data = img2.data * photflam / exptime / 0.0455**2 # get into units of erg/s/cm^2/A/arcsec^2


#Zoom into region of interest
cut_ctr = SkyCoord('12h18m57.5s 47d18m14s') 
cut_dims = np.array([4.0, 4.0]) * u.arcmin
cut = Cutout2D(img2.data, cut_ctr, cut_dims, wcs = img.wcs)

#Plot first subplot: raw data gathered by the telescope
plt.subplot(131, projection = img.wcs)
plt.imshow(cut.data, origin = 'lower', cmap = 'plasma')
plt.grid(color = 'yellow', ls = 'solid')
plt.title('Raw Telescope Data', weight = 'bold')
plt.ylabel('Declination (J2000)')

#Features of raw data are hard to see, so time to stretch the values
trans = viz.LogStretch() + viz.ManualInterval(0, 5e-19)
cut.data = trans(cut.data)

#Plot second subplot: Enhanced so all bright regions are more visible
plt.subplot(132, projection = img.wcs)
plt.imshow(cut.data, origin = 'lower', cmap = "plasma")
plt.grid(color = 'yellow', ls = 'solid')
plt.title('Enhanced', weight = 'bold')
plt.xlabel('Right Ascension (J2000)')


#Gaussian filter to supress point sources of light, like other stars
def destar(I, sigma, t):
    D = np.zeros_like(I)
    B = gauss(I, sigma)
    M = I - B
    for i in range(len(I)):
        for j in range(len(I[0])):
            if M[i][j] > t*I[i][j]:
                D[i][j] = B[i][j]
            else:
                D[i][j] = I[i][j]
    return D

#Apply that filter to the original image
cut2 = Cutout2D(img2.data, cut_ctr, cut_dims, wcs = img.wcs)
newcut = destar(cut2.data, 3, 0.5)
newcut = destar(newcut, 3, 0.5)
newcut = trans(newcut) #stretch these new values

# Plot third subplot: Enhanced and filtered image
plt.subplot(133, projection = img.wcs)
plt.imshow(newcut, origin = 'lower', cmap = "plasma")
plt.grid(color='yellow', ls='solid')
plt.title('Enhanced + Filtered', weight = 'bold')

plt.suptitle('M106', weight = 'bold', size = 16)
plt.show()