from PIL import Image
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

# sun data

img = Image.open("../data/sun_data/IMG_0956.CR2")
ifd = img.getexif().get_ifd(0x8769)
exif = dict(img.getexif().items())
members = 'Shahmari_Sani_Keyvanfar'
expname = 'Sun_Limb_Darkening'
isorate = 'ISO'+str(ifd[34855])
cammodel = exif[272].replace(" ", "_")
datetime = exif[306].split(" ")[0].replace(":", "")
exposure = str(float(round(ifd[33434], 5)))+'S'

filename = members+'-'+expname+'-'+isorate+'-' + \
    cammodel+'-'+datetime+'-'+exposure+'.fits'
metadata = {'Members': members, 'Trial': expname, 'ISO': isorate,
            'Model': cammodel, 'Data': datetime, 'Exposure': exposure}

hdr = fits.Header(metadata)
hdu = fits.PrimaryHDU(np.array(img.convert('L')), header=hdr)
hdu.writeto('../data/fits_data/'+filename, overwrite=True)

# dark data

ImageCollections = []

for rawfilename in os.listdir("../data/dark_data/"):
    img = Image.open("../data/dark_data/"+rawfilename)
    ImageCollections.append(np.array(img.convert('L')))

expname = 'Dark_Current'

xres, yres = ImageCollections[0].shape
SC_Collection = [sigma_clip([ImageCollections[i][x, y] for i in range(len(ImageCollections))]
                            ).data for x in range(xres) for y in range(yres)]
MedDark_IMG = np.median(SC_Collection, axis=1).reshape(xres, yres)
filename = members+'-'+expname+'-'+isorate+'-'+cammodel+'-'+datetime+'-'+exposure+'.fits'
metadata = {'Members': members, 'Trial': expname, 'ISO': isorate, 'Model': cammodel, 'Data': datetime, 'Exposure': exposure}
hdr = fits.Header(metadata)
hdu = fits.PrimaryHDU(MedDark_IMG, header=hdr)
hdu.writeto('../data/fits_data/'+filename, overwrite=True)

