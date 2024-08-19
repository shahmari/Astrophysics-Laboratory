from PIL import Image
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
figuredir = '../figs/'

hdul = fits.open("../data/fits_data/"+os.listdir("../data/fits_data")[0])
dark_img_array = hdul[0].data
hdul = fits.open("../data/fits_data/"+os.listdir("../data/fits_data")[1])
members = hdul[0].header['Members']
expname = hdul[0].header['Trial']
isorate = hdul[0].header['ISO']
cammodel = hdul[0].header['Model']
datetime = hdul[0].header['Data']
exposure = hdul[0].header['Exposure']
sun_img_array = hdul[0].data
corrected_sun = sun_img_array - dark_img_array

plt.imshow(corrected_sun, cmap='gist_heat')
plt.title('Sun in multi-channel')
plt.xlim(2000, 4000)
plt.ylim(200, 2200)
plt.colorbar()
plt.savefig(figuredir + 'sun_multi.png')
plt.show()

sun_pixels = np.argwhere(corrected_sun > 1000)
ymin_sun, xmin_sun = np.min(sun_pixels, axis=0)
ymax_sun, xmax_sun = np.max(sun_pixels, axis=0)
xdiameter, ydiameter = xmax_sun - xmin_sun, ymax_sun - ymin_sun
radius = round((xdiameter + ydiameter) / 4)
geometric_center = (xmin_sun + radius, ymin_sun + radius)
y_com, x_com = np.mean(sun_pixels, axis=0)
print("X_min:", xmin_sun)
print("X_max:", xmax_sun)
print("Y_min:", ymin_sun)
print("Y_max:", ymax_sun)
print("xdiameter:", xdiameter, ", ydiameter:", ydiameter)
print("radius:", radius)
print("geometric center (x, y):", geometric_center)
print("x com:", int(x_com), "y com:", int(y_com))

sun_grayscale_normalized = corrected_sun / corrected_sun.std()
intensity = sun_grayscale_normalized[int(y_com), int(x_com):xmax_sun]

plt.title("Horizontal Line, Going to the Right\n"
          "from the Center of the Sun")
plt.xlabel("$x$")
plt.ylabel("$I(x) / I(0)$")
plt.plot(intensity / intensity[0], color="mediumblue")
plt.grid()
plt.savefig(figuredir + 'intensity.png')
plt.show()

x = np.linspace(0, radius, len(intensity))
mu = np.sqrt(radius**2 - x**2) / radius
plt.title("Eddington Approximation")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$I(\mu) / I(0)$")

plt.plot(mu, intensity / intensity[0],
         color="mediumblue", label="Actual")
plt.plot(mu, (2 + 3 * mu) / 5, color="red",
         ls="dashed", label="Eddington")
# fitting a line on the intensity graph
def eddington(mu, a, b):
    return a + b * mu


popt, pcov = curve_fit(
    eddington, mu[int(len(mu)*0.4):], (intensity / intensity[0])[int(len(mu)*0.4):])
perr = np.sqrt(np.diag(pcov))
plt.plot(mu, eddington(mu, *popt), color="green", ls="dashed",
         label=r"Fit: $\frac{I(\mu)}{I(0)} = a + b\mu$" + f"\n$a = {popt[0]:.2f} \pm {perr[0]:.2f}$\n$b = {popt[1]:.2f} \pm {perr[1]:.2f}$")

plt.grid()
plt.legend()
plt.savefig(figuredir + 'eddington.png')
plt.show()
