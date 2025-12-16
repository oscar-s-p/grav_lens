from astropy.utils.data import download_file
from astropy.io import fits

import matplotlib.pyplot as plt

# List of FITS URLs for different filters
fits_urls = [
    "http://data.astropy.org/tutorials/FITS-images/M13_blue_0001.fits",
    "http://data.astropy.org/tutorials/FITS-images/M13_blue_0002.fits",
    "http://data.astropy.org/tutorials/FITS-images/M13_blue_0003.fits"
]
fig, axs = plt.subplots(1,3)
# Download files and open with astropy
for i, url in enumerate(fits_urls):
    file_path = download_file(url, cache=True)
    with fits.open(file_path) as hdul:
        data = hdul[0].data 
        # Do something with 'data', e.g., save or process it
        axs[i].imshow(data, origin = 'lower')

plt.show()
