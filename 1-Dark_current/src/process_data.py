from PIL import Image
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
figuredir = '../figures/'

# 13 Feb data

ImageCollections = dict()

for rawfilename in os.listdir("../raw-data/13Feb"):
    img = Image.open("../raw-data/13Feb/"+rawfilename)
    ifd = img.getexif().get_ifd(0x8769)
    exif = dict(img.getexif().items())
    members = 'Shahmari_Sani_Keyvanfar'
    expname = 'Dark_Current'
    isorate = 'ISO'+str(ifd[34855])
    cammodel = exif[272].replace(" ", "_")
    datetime = exif[306].split(" ")[0].replace(":", "")
    exposure = str(float(round(ifd[33434], 5)))+'S'

    # Convert the image to numpy array
    img = img.convert("L")
    img_array_SC = sigma_clip(np.array(img.convert('L'))).data

    if exposure not in ImageCollections:
        ImageCollections[exposure] = [
            {'Members': members, 'Trial': expname, 'ISO': isorate, 'Model': cammodel, 'Data': datetime, 'Exposure': exposure}, [img_array_SC]]
    else:
        ImageCollections[exposure][1].append(img_array_SC)


for key in ImageCollections.keys():
    avg_img = np.mean(ImageCollections[key][1], axis=0)
    members = ImageCollections[key][0]['Members']
    expname = ImageCollections[key][0]['Trial']
    isorate = ImageCollections[key][0]['ISO']
    cammodel = ImageCollections[key][0]['Model']
    datetime = ImageCollections[key][0]['Data']
    exposure = ImageCollections[key][0]['Exposure']
    filename = members+'-'+expname+'-'+isorate+'-'+cammodel+'-'+datetime+'-'+exposure+'.fits'
    metadata = {'Members': members, 'Trial': expname, 'ISO': isorate, 'Model': cammodel, 'Data': datetime, 'Exposure': exposure}

    hdr = fits.Header(metadata)
    hdu = fits.PrimaryHDU(avg_img, header=hdr)
    hdu.writeto('../fits-data/13Feb/'+filename, overwrite=True)

# 26 Feb data

ImageCollections = dict()

for rawfilename in os.listdir("../raw-data/26Feb"):
    img = Image.open("../raw-data/26Feb/"+rawfilename)
    ifd = img.getexif().get_ifd(0x8769)
    exif = dict(img.getexif().items())
    members = 'Shahmari_Sani_Keyvanfar'
    expname = 'Dark_Current'
    isorate = 'ISO'+str(ifd[34855])
    cammodel = exif[272].replace(" ", "_")
    datetime = exif[306].split(" ")[0].replace(":", "")
    exposure = str(float(round(ifd[33434], 5)))+'S'

    # Convert the image to numpy array
    img = img.convert("L")
    img_array_SC = sigma_clip(np.array(img.convert('L'))).data

    if exposure not in ImageCollections:
        ImageCollections[exposure] = [
            {'Members': members, 'Trial': expname, 'ISO': isorate, 'Model': cammodel, 'Data': datetime, 'Exposure': exposure}, [img_array_SC]]
    else:
        ImageCollections[exposure][1].append(img_array_SC)
        
for key in ImageCollections.keys():
    avg_img = np.mean(ImageCollections[key][1], axis=0)
    members = ImageCollections[key][0]['Members']
    expname = ImageCollections[key][0]['Trial']
    isorate = ImageCollections[key][0]['ISO']
    cammodel = ImageCollections[key][0]['Model']
    datetime = ImageCollections[key][0]['Data']
    exposure = ImageCollections[key][0]['Exposure']
    filename = members+'-'+expname+'-'+isorate+'-' + \
        cammodel+'-'+datetime+'-'+exposure+'.fits'
    metadata = {'Members': members, 'Trial': expname, 'ISO': isorate,
                'Model': cammodel, 'Data': datetime, 'Exposure': exposure}

    hdr = fits.Header(metadata)
    hdu = fits.PrimaryHDU(avg_img, header=hdr)
    hdu.writeto('../fits-data/26Feb/'+filename, overwrite=True)
