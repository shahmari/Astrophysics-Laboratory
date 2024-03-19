from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
figuredir = '../figures/'

# 13 Feb data

conv_str_exp = {'0.03333S': '1/30 s', '0.2S': '1/5 s', '1.0S': '1 s', '10.0S': '10 s', '30.0S': '30 s'}  # Exposure time conversion (For the labels of the histogram)
range_exp = {'0.03333S': 0.4, '0.2S': 0.4, '1.0S': 0.4, '10.0S': 0.4, '30.0S': 0.8}  # Range of the exposure time (For the histogram)
strings_to_float = {'0.03333S': 1/30, '0.2S': 0.2, '1.0S': 1, '10.0S': 10, '30.0S': 30}

exposures = []
medians = []
sigmas = []

for rawfilename in os.listdir("../fits-data/13Feb"):
    hdul = fits.open("../fits-data/13Feb/"+rawfilename)
    img_array = hdul[0].data
    members = hdul[0].header['Members']
    expname = hdul[0].header['Trial']
    isorate = hdul[0].header['ISO']
    cammodel = hdul[0].header['Model']
    datetime = hdul[0].header['Data']
    exposure = hdul[0].header['Exposure']
    
    exposures.append(strings_to_float[exposure])
    medians.append(np.median(img_array))
    sigmas.append(np.std(img_array))

    filename = figuredir + '13Feb-PixelValueDistribution-' + \
        isorate + '-' + \
        conv_str_exp[exposure].replace(' ', '').replace('/', '_')+'.png'
    plt.figure()
    plt.hist(np.mean(img_array, axis=0).reshape(
        -1), bins=33, range=(0, range_exp[exposure]), color='blue', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Pixel Value Distribution, exposure time: ' +
              conv_str_exp[exposure])
    plt.savefig(filename)
    plt.close()
    
plt.figure()
plt.plot(np.log10(exposures), medians, 'o')
plt.xlabel('log10(Exposure time) (s)')
plt.ylabel('Median pixel value')
plt.title('Median pixel value vs. exposure time')
plt.savefig(figuredir+'13Feb-MedianPixelValueVsExposureTime.png')
plt.close()

plt.figure()
plt.plot(np.log10(exposures), sigmas, 'o')
plt.xlabel('log10(Exposure time) (s)')
plt.ylabel('Standard deviation of pixel value')
plt.title('Standard deviation of pixel value vs. exposure time')
plt.savefig(figuredir+'13Feb-StdPixelValueVsExposureTime.png')
plt.close()

# 26 Feb data

conv_str_exp = {'0.00025S': '1/4000 s', '0.001S': '1/1000 s', '0.01S': '1/100 s', '0.1S': '1/10 s',
                '1.0S': '1 s', '10.0S': '10 s', '30.0S': '30 s'}  # Exposure time conversion (For the labels of the histogram)
strings_to_float = {'0.00025S': 1/4000, '0.001S': 1/1000, '0.01S': 1/100, '0.1S': 1/10, '1.0S': 1, '10.0S': 10, '30.0S': 30}

exposures = []
medians = []
sigmas = []

for rawfilename in os.listdir("../fits-data/26Feb"):
    hdul = fits.open("../fits-data/26Feb/"+rawfilename)
    img_array = hdul[0].data
    members = hdul[0].header['Members']
    expname = hdul[0].header['Trial']
    isorate = hdul[0].header['ISO']
    cammodel = hdul[0].header['Model']
    datetime = hdul[0].header['Data']
    exposure = hdul[0].header['Exposure']
    
    exposures.append(strings_to_float[exposure])
    medians.append(np.median(img_array))
    sigmas.append(np.std(img_array))
    
    filename = figuredir + '26Feb-PixelValueDistribution-' + \
        isorate + '-' + \
        conv_str_exp[exposure].replace(' ', '').replace('/', '_')+'.png'
    plt.figure()
    plt.hist(np.mean(img_array, axis=0).reshape(
        -1), bins=20, range=(0, 0.2), color='blue', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Pixel Value Distribution, exposure time: ' +
              conv_str_exp[exposure])
    plt.savefig(filename)
    plt.close()
    
plt.figure()
plt.plot(np.log10(exposures), medians, 'o')
plt.xlabel('log10(Exposure time) (s)')
plt.ylabel('Median pixel value')
plt.title('Median pixel value vs. exposure time')
plt.savefig(figuredir+'26Feb-MedianPixelValueVsExposureTime.png')
plt.close()

plt.figure()
plt.plot(np.log10(exposures), sigmas, 'o')
plt.xlabel('log10(Exposure time) (s)')
plt.ylabel('Standard deviation of pixel value')
plt.title('Standard deviation of pixel value vs. exposure time')
plt.savefig(figuredir+'26Feb-StdPixelValueVsExposureTime.png')
plt.close()
