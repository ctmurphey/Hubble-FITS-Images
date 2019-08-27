import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.nddata import NDData, NDDataRef, StdDevUncertainty, CCDData, Cutout2D
from astropy.coordinates import SkyCoord
import astropy.visualization as viz
from scipy.ndimage import gaussian_filter as gauss

hdulist = fits.open('ngc6212.fits')

photflam = hdulist[1].header['PHOTFLAM']
photzpt = hdulist[1].header['PHOTZPT']
exptime = hdulist[0].header['EXPTIME']

data = hdulist[1].data
newdat = data * photflam / exptime / 0.0455**2

img = CCDData.read('ngc6212.fits', unit='adu')
img2 = img.copy()
img2.data = img2.data * photflam / exptime / 0.0455**2
cut_ctr = SkyCoord('16h43m23.111s 39d48m22.829s')
cut_dims = np.array([0.5, 0.5]) * u.arcmin
cut = Cutout2D(img2.data, cut_ctr, cut_dims, wcs = img.wcs)

plt.subplot(131, projection = img.wcs)
plt.imshow(cut.data, origin = 'lower', cmap = 'inferno')
plt.grid(color='yellow', ls='solid')
plt.title("Raw Data", weight = 'bold')
plt.ylabel('Declination (J2000)', weight = 'bold')

trans = viz.LogStretch() + viz.ManualInterval(0, 7e-16)
cut.data = trans(cut.data)
plt.subplot(132, projection = img.wcs)
plt.imshow(cut.data, origin = 'lower', cmap = "inferno")
plt.grid(color='yellow', ls='solid')
plt.xlabel('Right ascension (J2000)', weight = 'bold')
plt.title('Enhanced', weight = 'bold')

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

cut2 = Cutout2D(img2.data, cut_ctr, cut_dims, wcs = img.wcs)
newcut = destar(cut2.data, 3, 0.25)
newcut = destar(newcut, 3, 0.25)
newcut = destar(newcut, 3, 0.25)
newcut = trans(newcut)
plt.subplot(133, projection = img.wcs)
plt.imshow(newcut, origin = 'lower', cmap = "inferno")
plt.grid(color='yellow', ls='solid')
plt.title('Enhanced and Filtered', weight = 'bold')

plt.suptitle('NGC 6212', weight = 'bold', size = 16)
plt.show()