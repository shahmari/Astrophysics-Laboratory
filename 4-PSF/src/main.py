from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import curve_fit
from scipy.ndimage import label, center_of_mass
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
figuredir = '../figs/'

# Loading the fits data
hdul = fits.open("../data/"+os.listdir("../data/")[1])

# Label connected regions in the mask
mask = hdul[0].data > 400
labels, num_stars = label(mask)

# counting the number of pixels in each star
sizes = np.bincount(labels.ravel())

universal_star_centers = []
local_star_centers = []
star_frames = []

# Iterate over each labeled region (star)
for i in range(1, num_stars + 1):
    if sizes[i] > 100:
        # Get the indices of pixels belonging to the current star
        star_indices = np.where(labels == i)

        # Append the frame of the star to the list
        lower_bound = star_indices[0].min()
        upper_bound = star_indices[0].max()
        left_bound = star_indices[1].min()
        right_bound = star_indices[1].max()
        star_frame = hdul[0].data[lower_bound:upper_bound,
                                  left_bound:right_bound]
        star_frames.append(star_frame)

        # Calculate the center of mass for the current star
        local_star_center = center_of_mass(
            star_frame, labels[lower_bound:upper_bound, left_bound:right_bound], index=i)
        universal_star_center = center_of_mass(hdul[0].data, labels, index=i)

        # Append the center of mass to the list
        local_star_centers.append(local_star_center)
        universal_star_centers.append(universal_star_center)


def gaussian(x, amplitude, stddev):
    return amplitude * np.exp((-(x / stddev) ** 2) / 2)


def fit_gaussian(star_frame, star_center):
    # distance from the center of the star
    x = np.arange(star_frame.shape[1])
    y = np.arange(star_frame.shape[0])
    x, y = np.meshgrid(x, y)
    r = np.sqrt((x - star_center[1]) ** 2 + (y - star_center[0]) ** 2)
    r = r.astype(int)
    # average intensity of the star at each distance from the center
    intensity = np.bincount(r.ravel(), star_frame.ravel()
                            ) / np.bincount(r.ravel())
    # fit a gaussian to the intensity profile
    popt, perr = curve_fit(gaussian, np.arange(
        len(intensity)), intensity, p0=[1, 1])
    return intensity, popt, perr

intensities = []
popts = []
perrs = []
for i in range(len(star_frames)):
    star_frame = star_frames[i]
    star_center = local_star_centers[i]
    intensity, popt, perr = fit_gaussian(star_frame, star_center)
    intensities.append(intensity)
    popts.append(popt)
    perrs.append(np.sqrt(np.diag(perr)))

for i in range(len(star_frames)):
    intensity = intensities[i]
    popt = popts[i]
    perr = perrs[i]
    star_frame = star_frames[i]
    star_center = local_star_centers[i]
    star_radius = popt[1]

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(intensity / max(intensity), label='Intensity profile')
    axs[0].plot(gaussian(np.arange(len(intensity)), *popt) /
                popt[0], label='Gaussian fit')
    axs[0].grid()
    axs[0].legend()
    # putting the text on left center of the plot
    axs[0].text(0.05, 1.05*min(intensity / max(intensity)),
                f"Amplitude: {popt[0]:.2f}" + r"$\pm$" + f" {perr[0]:.2f}\n" + f"Standard deviation: {np.abs(popt[1]):.2f}" + r"$\pm$" + f" {np.abs(perr[1]):.2f}", fontsize=8)
    axs[0].set_xlabel('Distance from the center')
    axs[0].set_ylabel('Intensity (Scaled)')
    axs[1].imshow(star_frame, cmap='gray')
    axs[1].scatter(*star_center[::-1], color='red', marker='x')
    axs[1].add_patch(patches.Circle(
        star_center[::-1], star_radius, color="blue", fill=False, lw=2))
    axs[1].figure.set_size_inches(10, 5)
    axs[1].set_aspect('equal')
    axs[1].title.set_text('Star frame')
    axs[1].axis('off')
    plt.savefig(figuredir + f'star_{i}.png')
    plt.close()

sigmas = np.sort(np.abs(np.array(popts)[:, 1]))[0:-2]
plt.hist(sigmas, bins=20, edgecolor="black")
plt.title("Distribution of standard deviations of the Gaussian fits")
plt.text(
    11, 11, f"Median: {np.median(sigmas):.2f}, Standard deviation: {np.std(sigmas):.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Standard deviation")
plt.ylabel("Frequency")
plt.savefig(figuredir+"histogram.png")
plt.show()