### RGB test ###

from image_viewer_class import image_viewer
import numpy as np
import astropy
import matplotlib.pyplot as plt

def manual_norm(data, min, max):
    d = (data-min)/(max-min)
    d_mask = d < 0
    d[d_mask] = 0
    return d

iv = image_viewer('test_images', folder_list=['2025-11-11'], list_available=False)
obj = iv.df_files['object'].loc[0]
obj_int = iv.df_grav_lens.index[iv.df_grav_lens['object'] == obj].tolist()[0]
obj = iv.df_grav_lens['object'].loc[obj_int]
ra = iv.df_grav_lens['ra'].loc[obj_int]
dec = iv.df_grav_lens['dec'].loc[obj_int]


data = [iv.read_data(0), iv.read_data(1),
        iv.read_data(2), iv.read_data(3)]
heads = [iv.read_data(0, header=True), iv.read_data(1, header=True),
         iv.read_data(2, header=True), iv.read_data(3, header=True)]
sky_flux = [h['FLUXSKY'] for h in heads]
data_norm = []
d_norm = []
d_cut = []


for i in range(3):
    im_str, im = iv.return_index(i) # type: ignore
    d_cut = iv.data_manipulation(im_str, centered = (ra, dec), # type: ignore
                                 zoom = '0 3 1 d')[0] # type: ignore
    data[i] = d_cut.data
    if i==0:
        fig, ax = plt.subplots(2,2, figsize = (14,8), subplot_kw=dict(projection=d_cut.wcs))
        ax = ax.ravel()
    j = 3*i
    norm_i = astropy.visualization.simple_norm(data[i],
                                               stretch = 'linear',
                                               vmin = sky_flux[i],
                                               max_percent = 100,
                                               clip = True
                                               )
    manual_min = sky_flux[i]
    manual_max = sky_flux[i]*1.1
    print(manual_min, manual_max)
    d_norm.append(manual_norm(data[i], manual_min, manual_max))

    ax[i].imshow(d_norm[i], origin = 'lower')
    ax[i].set_title(str(iv.df_files['filter'].loc[i]) + str(sky_flux[i]))



rgb = astropy.visualization.make_lupton_rgb(d_norm[2], d_norm[1], d_norm[0],
                                            stretch = 1,
                                            Q = 8)

ax[-1].imshow(rgb, origin = 'lower')
plt.tight_layout()
plt.show()