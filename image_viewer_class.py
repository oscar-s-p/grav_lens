'''
Image viewer class
'''

import os, sys, glob
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %matplotlib qt

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
# from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.visualization import SimpleNorm, simple_norm
from astropy.visualization import make_lupton_rgb
from astropy.visualization.wcsaxes import add_scalebar
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.coordinates import get_body, EarthLocation
from astropy.nddata import Cutout2D
from astropy.time import Time
from math import ceil


class image_viewer:
    def __init__(self, directory: str = '',
               list_available = False,
               folder_list = [],
               previous_df = False,
               print_error = True):
        """Class to quickly open FITS images. Searches in given directory.
        
        Attributes
        ---------
        directory : str
            Directory where images are stored. If none given look in current working directory.

        list_available : bool : False
            Wether to print the resulting dataframe of found images or not

        folder_list : optional, list of str
            Extra directories to inspect for images and save their folder path from working directory
        
        previous_df : optional, pd.DataFrame or str
            Previous df with files to be added to the new one of the files found in ``folder_list``

        print_error: bool, optional
            If False, no error warnings will be printed (good for reading large datasets)
        
        Methods
        --------
        return_index()
            Returns the image path and index in the datafile given one or the other.
        
        header_info()
            Method to view general header info.

        view()
            Method to view images.
        
        view_multiple()
            Method to view multiple images in subplots of a figure
        """
        self.folder_list = folder_list
        print('Current working directory: ' + os.getcwd())
        if directory=='':
            directory = os.getcwd()
        if directory != os.getcwd():
            self.dir_img = os.path.join(os.getcwd(),directory)
        else: self.dir_img = directory
        print('Image directory defined: ' + self.dir_img)

        # list of images in dir_img and where were they
        files = list(Path(self.dir_img).glob('*.fits'))
        folder_found = ['']*len(files)
        # list of images in the different folders of folder_list and the corresponding folder
        if folder_list!= []:
            for fl in folder_list:
                fi = list(Path(os.path.join(self.dir_img, fl)).glob('*.fits'))
                files=files+fi
                folder_found =folder_found+[fl]*len(fi)

        files_data = []
        # creation of data dictionary
        for k, f in enumerate(files):
            try:
                name = f.name
                path = str(f.resolve())
                try: telescope, camera, date_time, object, filter = name.split('_')
                except: 
                    if print_error: print('ERROR WITH FILENAME FORMAT CONVENTION EXPECTED')
                size_MB = f.stat().st_size / 1e6
                created = pd.to_datetime(f.stat().st_ctime, unit="s")
                files_data.append({"filename": name, "path": path, "telescope": telescope, 'camera': camera,
                                   "object": object, "filter": filter[:-5], "size_MB": size_MB,
                                   "date_time": pd.to_datetime(date_time, format='%Y-%m-%d-%H-%M-%S-%f'),
                                   "folder_found": folder_found[k]})
            except: 
                if print_error: print('Error with file: %s'%f)
                
        if len(files)==0:
            print('WARNING: NO IMAGE FILES FOUND')
            return
        # creation of dataframe
        df_files = pd.DataFrame(files_data).sort_values("filename").reset_index(drop=True)
        # Addition of previous dataframe
        if type(previous_df) != bool:
            if type(previous_df) != pd.DataFrame:
                if type(previous_df) == str:
                    if previous_df[-3:] == 'pkl': previous_df = pd.read_pickle(previous_df)
                    elif previous_df[-3:] == 'csv' : previous_df = pd.read_csv(previous_df)
                    else: 
                        print('ERROR: unrecognized DataFrame format. Use \'.pkl\' or \'.csv\'.')
                        return
            self.df_files = pd.concat([df_files, previous_df], ignore_index = True).drop_duplicates(subset = 'filename', keep= 'last')
        else: self.df_files = df_files
        # print available images if requested
        if list_available:
            print(self.df_files)
        print('Total number of images found: ', len(self.df_files))

        # Store gravitational lens objects
        grav_lens = ['QSO0957+561', 'Q2237+030', 'MG1654+1346', 'SDSSJ1004+4112', 'LBQS1333+0113', 'SDSSJ0819+5356',
             'EinsteinCross', 'DESI-350.3458-03.5082', 'ZTF25abnjznp']
        # EinsteinCross and Q2237+030 are the same object (?)
        grav_lens_ra = ['10 01 20.692 h', '22 40 30.234 h', '16 54 41.796 h', '10 04 34.936 h', '13:35:34.8 h', '08 19 59.764 h',
                        '22 40 30.271 h', '350.3458d', '07:16:34.5h']
        grav_lens_dec = ['+55 53 55.59 d', '+03 21 30.63 d', '+13 46 21.34 d', '+41 12 42.66 d', '+01 18 05.5 d', '+53 56 24.63 d',
                         '+03 21 31.03 d', '-03.5082d', '+38:21:08d']
        grav_data = []
        for i in range(len(grav_lens)):
            grav_data.append({
                'object' : grav_lens[i],
                'ra' : Angle(grav_lens_ra[i]),
                'dec' : Angle(grav_lens_dec[i])
                })
        self.df_grav_lens = pd.DataFrame(grav_data).sort_values('object').reset_index(drop=True)
    
    def return_index(self, image):
        """
        Returns the image path and index in the datafile given one or the other.

        Parameters
        ----------
        image: int / str
            int - image index in datafile \n
            str - image path
        """
        if type(image)==int:
            image_str = self.df_files.loc[image].filename
            image_int = image
        else: 
            image_str = image
            try: image_int = self.df_files.index[self.df_files['filename']==image].to_list()[0]
            except:
                print('\n ERROR: FILENAME NOT FOUND')
                return
        if self.folder_list != False:
            folder_name = self.df_files.iloc[image_int].folder_found
            image_str = os.path.join(folder_name, image_str) # type: ignore
        return image_str, image_int
    

    def image_finder(self, object, 
                     date = None, 
                     filter = None,
                     return_df = False,
                     printeo = False
                     ):
        """
        Method to identify the fits file that match an observation object, date and filter.
        
        Parameters
        ----------
        object : index / str
            Either the iloc or string to the object in self.df_grav_lens

        date : 'YYYY-MM-DD' (optional hh-mm-ss)
            If no date is supplied, return possible options

        filter : str
            Desired filter. If None, return possible options
        """
        try:
            if type(object) == str:
                obj_int = self.df_grav_lens.index[self.df_grav_lens['object'] == object].tolist()
                if obj_int == []:
                    print('ERROR: OBJECT NAME NOT REGISTERED.\n  Try with one of: ', self.df_grav_lens['object'].tolist())
                obj_str = object
            if type(object) == int:
                obj_str = self.df_grav_lens['object'].iloc[object]
                obj_int = object
        except: 
            print('ERROR: No previously known object was found.\n  Try with one of: ', self.df_grav_lens['object'].tolist())
        
        df_filtered = self.df_files[self.df_files["object"]==obj_str].copy()

        if date == None:
            print('Available date observations:')
            print(df_filtered.groupby(['object', 'folder_found']).size())
        
        if date != None:
            if type(date) == str:
                df_filtered = df_filtered[df_filtered['folder_found'] == date]
            if type(date) == list:
                df_filtered = df_filtered[df_filtered['folder_found'] in date]

        if filter != None:
            df_filtered = df_filtered[df_filtered['filter'] == filter]
        
        if return_df == True:
            print('Matching index: ')
            print(df_filtered.index.tolist())
            return df_filtered
        else:
            return df_filtered.index.tolist()


    def header_info(self, image,
                    interesting_keys = ['INSTRUME', 'OBJECT', 'FILTER', 'INTEGT', 'DATE-OBS',
                                        'RA', 'DEC', 'NAXIS1', 'NAXIS2', 'SCALE', 'FOVX', 'FOVY',
                                        'CCW', 'CRPIX1', 'CRPIX2', 'FWHM']
                                        ):
        """Method to view general header info.
        
        Parameters
        ----------
        image : int / str
            int - index of desired file in dataframe \n
            string - path to desired fits file
            
        interesting_keys: list / 'all'
            list - list of strings with header keyword \n
            'all' - will print the whole header
        """
        image_str, image_int = self.return_index(image) # type: ignore
        
        # Extracting data from header
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            heads = hdul[0].header # type: ignore
            hdul.close()
        # printing basic header info
        print('Image: %s'%image_str)
        print('\n   --- HEADER DATA ---')
        if type(interesting_keys) == str and interesting_keys!='all':
                interesting_keys = [interesting_keys]
        try:
            if type(interesting_keys) == str and interesting_keys=='all':
                print(repr(heads))
            else:
                for k in interesting_keys:
                    if heads.comments[k]!='':
                        print(k, ' = ', heads[k], '  ---  ', heads.comments[k])
                    else:
                        print(k, ' = ', heads[k])
        except:
            print('WARNING: WRONG interesting_keys PARAMETER.')
            print('         Header parameter not recognized. Try the string \'all\' to view the full header')

    def view_image(self, image,
                    RGB = False,
                    nrows_ncols = None,
                    figsize = None,
                    manipulation_kw = {
                       'centered' : True,
                       'zoom' : False,
                       'stretch' : 'linear',
                       'percentile' : None,
                       'vminmax' : (None, None)
                       },
                    plotting_kw = {
                        'cmap' : 'gray',
                        'scalebar_arcsec' : 5,
                        'scalebar_frame' : False,
                        'add_circle' : None
                        },
                    RGB_kw = {
                        'stretch' : 5,
                        'Q' : 8,
                        'minimum' : None
                        },
                    RGB_norm_kw = {
                        'vmax' : None,
                        'max_percentile' : 99,
                        'max_sky' : False
                        }
                    ):
        """
        Method to view images. Takes dictionary keywords for ``data_manipulation`` and ``plotting``.
        """
        # Multiple images
        if type(image) == list and RGB == False:
            print('------\nViewing multiple images:')
            n_image = len(image)
            if nrows_ncols == None:
                if n_image <= 3: nrows_ncols = (1, n_image)
                else: nrows_ncols = (ceil(np.sqrt(n_image)), ceil(np.sqrt(n_image)))
            image_list = image

        # Simple image Non RGB
        if type(image) != list:
            print('------\nViewing image:')
            n_image, nrows_ncols = 1, (1,1)
            image_list = [image]
        # RGB image
        if RGB == True: 
            n_image = 1
            colors = ['R', 'G', 'B']
            cutout_RGB = []
            print('------\nRGB color composite image:')
            if n_image == 1: nrows_ncols = (1,1)
            image_list = image

        self.nr_nc = nrows_ncols
        n_data = len(image_list)

        # if manipulation and plotting are dicts, use the same setup for all images
        if type(manipulation_kw) == dict: manipulation_kw = [manipulation_kw]*n_data
        if type(plotting_kw) == dict: plotting_kw = [plotting_kw]*n_data

        fig, axes = plt.subplots(self.nr_nc[0], self.nr_nc[1], # type: ignore
                                 figsize = figsize)
        if n_image == 1: axes = [axes]
        axes = np.array(axes).reshape(-1)
        
        for i, (img, m_k, p_k) in enumerate(zip(image_list, manipulation_kw, plotting_kw)):
            self.img_str, self.img_int = self.return_index(img) # type: ignore
            cutout, norm = self.data_manipulation(self.img_str, **m_k) # type: ignore

            if RGB == False:
                print('    Object: ',self.df_files.object.loc[self.img_int],
                      '  -  Filter: ',self.df_files['filter'].loc[self.img_int])
                self.plotting(cutout, norm, fig, axes[i], i,
                              **p_k)
            else:
                # Extracting data from header
                with fits.open(os.path.join(self.dir_img, self.img_str)) as hdul: # type: ignore
                    heads = hdul[0].header # type: ignore
                    hdul.close()
                if i==0: print('    Object: ', self.df_files.loc[self.img_int].object)
                print('    - ',colors[i],': ', self.df_files['filter'].loc[self.img_int])
                # min and max for manual norm, if max_sky is set, use it to obtain max as max_sky * sky_flux
                vmin = heads['FLUXSKY']
                if 'vmax' in RGB_norm_kw.keys(): vmax = RGB_norm_kw['vmax']
                if RGB_norm_kw['max_sky'] != False: vmax = RGB_norm_kw['max_sky']*heads['FLUXSKY']
                if vmax == None: vmax = np.max(cutout.data)
                # manual normalization
                data = (cutout.data - vmin)/(vmax-vmin)
                data_mask = data < 1e-3
                data[data_mask] = 1e-3
                cutout_RGB.append(data)

                if i == len(image_list)-1:                        
                    rgb_default = make_lupton_rgb(cutout_RGB[0].data, cutout_RGB[1].data, cutout_RGB[2].data,
                                                  **RGB_kw)
                    self.plotting(cutout, norm, fig, axes[0],0,
                                  RGB = True, rgb_data = rgb_default,
                                  **plotting_kw[i])
        plt.tight_layout()
        plt.show()

    def data_manipulation(self, image_str,
                          centered = True, 
                          zoom = False,
                          stretch = 'linear',
                          percentile = None,
                          vminmax = (None, None)
                          ):
        """
        Method to prepare images for manipulation. It is internally called. Crops the image and sets visualization normalization and stretch.

        Parameters
        ---------
        image : int / string / list
            int - index of desired file in dataframe \n
            string - path to desired fits file \n

        centered : True or tuple, optional
            (x,y) - int for pix coordinates \n
            (RA, DEC) - wcs coordinates. Accepting both strings or angle values

        zoom : False or Value or Tuple, optional
            int / (int, int) - pixel size in x and y axis \n
            Angle / (Angle, Angle) - angular size in RA and DEC
        
        stretch : str, optional
            Image stretch to enhance detail visualization \n
            ``linear``, ``sqrt``, ``power``, ``log``, ``sinh``, ``asinh``
        
        percentile : int or tuple, optional
            ``int`` - Middle percentile of values to consider for normalization; 
            ``tuple`` - Lower and upper percentile of values to consider for normalization
        
        vminmax : tuple, optional
            Min and max pixel values for normalization. Overrides ``percentile``.
            If set as None, keeps the absolute min or max of image
        """
        

        # Extracting data from header
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul:
            data = hdul[0].data.astype(np.float32) # type: ignore
            heads = hdul[0].header # type: ignore
            wcs = WCS(heads)
            hdul.close()
        
        # obtaining central px coordinates
        x_shape = data.shape[1]
        y_shape = data.shape[0]
        if centered == True:
            center_px = (x_shape//2, y_shape//2)
        if type(centered)==tuple:
            if type(centered[0]) == int: # input in px units
                center_px = tuple(centered)
            elif type(centered[0]) == str: # input in str to be converted to deg
                center_angle = SkyCoord(centered[0], centered[1], frame = 'icrs')
                center_px = skycoord_to_pixel(center_angle, wcs, origin=0)
            else:
                center_angle = SkyCoord(centered[0], centered[1], frame = 'icrs')
                center_px = skycoord_to_pixel(center_angle, wcs, origin=0)
        
        # setting zoom
        if zoom == False:
            zoom = (x_shape, y_shape)
        if type(zoom) == str:
            zoom = Angle(zoom)
        if type(zoom)== tuple:
            if type(zoom[0]) == str:
                zoom = (Angle(zoom[0]), Angle(zoom[1]))
        if type(zoom)==tuple:
            zoom = zoom[::-1]
        
        # slicing image
        try:
            cutout = Cutout2D(data, position = center_px, size = zoom, wcs = wcs)
        except:
            print('\n --- \nERROR: the cutout region is outside of the image.')
            return

        # norm definition
        if type(percentile) == int or percentile == None:
            percentile_minmax = (None, None)
        if type(percentile) == tuple:
            percentile_minmax = percentile
            percentile = None
        if stretch not in {'linear', 'sqrt', 'power', 'log', 'sinh', 'asinh'}:
            print('ERROR: Stretch should be one of \'linear\', \'sqrt\', \'power\', \'log\', \'sinh\', \'asinh\'')
            plt.close()
            return
        norm = simple_norm(cutout.data, stretch = stretch, 
                           vmin = vminmax[0], vmax = vminmax[1],
                           percent = percentile,
                           min_percent = percentile_minmax[0],
                           max_percent = percentile_minmax[1])
        
        return cutout, norm
        
    def plotting(self,
                 cutout, norm, fig, ax, ax_i,
                cmap = 'gray',
                scalebar_arcsec = 5, scalebar_frame = False,
                add_circle = None,
                RGB = False,
                rgb_data = False
                ):
        """
        Method to plot images, obtains edited data from ``self.data_manipulation()``.

        Parameters
        ---------
        cutout : Cutout2D
            Selected cutout object from ``data_manipulation``

        norm : Norm
            Selected norm from ``data_manipulation``

        cmap : str, optional
            Select the desired colormap for the image

        scalebar_arcsec : int, optional
            Angular size of scalebar in arcsec units
        
        scalebar_frame : bool, optional
            Add frame or not

        add_circle : dict, list of dicts or None, optional
            Parameters to plot a circle overlay. If None, no circle is plotted. If multiple circles are desired, enter a list of dicts.\n
            Expected keys: \n
                'center' : tuple 
                    (RA, DEC) coordinates as astropy Angle or SkyCoord
                'size' : astropy.units.Quantity
                    Angular size (e.g., astropy Angle with units).
                'color' : str, optional
                    Circle edge color.
                'label' : str, optional
                    Label for the circle to use in legend.
            
        fig_kwrds : None or dict, optional
            Dict with all the keywords desired to insert in ``plt.subplots()``

        figure : None or dict ..... tuple or axis
            Dict used by view_multiple method. Expected keys: \n
                'is_simple' : bool
                'create_fig' : bool
                    True or False
                'figsize' : tuple
                    Looked at if ``create_fig = True``
                'nrows_ncols' : tuple
                    Looked at if ``create_fig = True``
                'fig' : plt.figure object
                    Looked at if ``create_fig = False``
                'ax' : plt.axis object
                    Looked at if ``create_fig = False``
                'im_i' : int
                    Subplot index (image index). Looked at if ``create_fig = False``

            None - creates normal figure, does not return nothing \n
            tuple (int, int) - creates figure with specified conditions. Returns (fig, ax) \n
            tuple (ax, int, int) - plots image in specified ax[int,int]
        """
        with fits.open(os.path.join(self.dir_img, self.img_str)) as hdul: # type: ignore
            heads = hdul[0].header # type: ignore
            hdul.close()
        ax.remove()
        ax = fig.add_subplot(self.nr_nc[0], self.nr_nc[1], ax_i+1, projection = cutout.wcs) # type: ignore
        if RGB == False:
        # colorbar
            cax = ax.imshow(cutout.data,
                            norm = norm, origin = 'lower',
                            cmap = cmap)
            cbar = plt.colorbar(cax)
            cbar.set_label('ADU', rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=10)
        else:
            ax.imshow(rgb_data, origin = 'lower')

        # Scale bar choosing color depending on luminance of cmap
        scalebar_angle = scalebar_arcsec/3600*u.deg # type: ignore
        rgba = plt.get_cmap(cmap)(0.0)
        luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        scalebar_color = 'white' if (luminance < 0.5 and scalebar_frame == False) else 'black'
        add_scalebar(ax, scalebar_angle, label="%s arcsec"%str(scalebar_arcsec), color=scalebar_color, frame=scalebar_frame)
        # Axis and title
        ax.set(xlabel='RA', ylabel='DEC')
        ax.coords.grid(color='gray', alpha=0.5, linestyle='solid')
        title_str = (r'$\bf{Object}$: %s - $\bf{Telescope}$: %s - $\bf{Seeing}$: %.1f$^{\prime\prime}$''\n'
                    r'$\bf{Camera}$: %s - $\bf{Filter}$: %s - $\bf{Integration}$: %s s''\n'
                    r'$\bf{SNR}$: %s - $\bf{Date time}$: %s - $\bf{Moon D}$: %.1fº'
                    %(self.df_files.iloc[self.img_int]['object'],
                    self.df_files.iloc[self.img_int]['telescope'],
                    (float(heads['FWHM'])*float(heads['SCALE'])),
                    self.df_files.iloc[self.img_int]['camera'],
                    self.df_files.iloc[self.img_int]['filter'],
                    heads['INTEGT'], heads['OBJECSNR'],
                    self.df_files.iloc[self.img_int]['date_time'].strftime("%Y-%m-%d %H:%M"),
                    self.get_moon_distance(self.img_int).deg))
        ax.set_title(title_str)
        ax.minorticks_on()

        # Optional plot of circles
        if add_circle is not None:
            if type(add_circle) != list:
                add_circle = [add_circle]
            for d_circle in add_circle:
                center = d_circle.get('center')
                size = d_circle.get('size')
                color = d_circle.get('color')
                label = d_circle.get('label')
                c = SphericalCircle((Angle(center[0]), Angle(center[1])),
                                    Angle(size),
                                    edgecolor = color,
                                    facecolor = 'none',
                                    transform = ax.get_transform('icrs'))
                ax.add_patch(c)
 

    def read_data(self, image, header = False):
        """Method to view images."""
        image_str, image_int = self.return_index(image) # type: ignore
        print('Reading ', image_str)

        # Extracting data from header
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            data = hdul[0].data.astype(np.float32) # type: ignore
            head = hdul[0].header # type: ignore
            hdul.close()

        if header == False: return data
        else: return head
    
    def get_moon_distance(self, image):
        """
        Method to calculate the angular separation with the Moon in degrees for a given observation.
        
        Parameters
        ---------
        image : int or str
            index of observation image or string to .fits file.
            
        Returns:
            astropy.Angle object with angular separation"""

        image_str, image_int = self.return_index(image) # type: ignore
        with fits.open(os.path.join(self.dir_img, image_str)) as hdul: # type: ignore
            heads = hdul[0].header # type: ignore
            hdul.close()
        RA = str(heads['RA']) + ' d'
        DEC = str(heads['DEC']) + ' d'
        time = Time(self.df_files.iloc[image_int]['date_time'])
        loc = EarthLocation.of_site('Observatorio del Teide')
        moon_coords = get_body('moon', time = time, location = loc)
        moon_coords = SkyCoord(ra = moon_coords.ra, dec = moon_coords.dec, frame = 'icrs', unit = u.deg) # type: ignore
        obj_coords = SkyCoord(ra = Angle(RA), dec = Angle(DEC), frame = 'icrs', unit = u.deg) # type: ignore
        sep = obj_coords.separation(moon_coords)
        return sep


