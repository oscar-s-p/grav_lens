# Image reading test file

import os, sys
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt
from astropy.io import fits

wd = os.getcwd()
directory = 'test_images'
print('Working directory: '+ wd)
print('Directory containing images: '+ directory)
print('Available images:')
dir_img = os.path.join(wd, directory)
list_images = os.listdir(dir_img)
print(list_images)

image_file = list_images[0]
image_data = fits.getdata(os.path.join(dir_img,image_file))

print('Image data type: ', type(image_data))
print('Image data shape: ', image_data.shape)

plt.close(1)
fig, ax = plt.subplots(num = 1)

ax.imshow(image_data[:1000,:1000])
plt.show() 