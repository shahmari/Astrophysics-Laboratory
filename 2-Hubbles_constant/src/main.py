import os
import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

os.chdir(os.path.dirname(__file__))
figuredir = '../figs/'
datadir = '../data/'

cepheids = pd.read_csv(datadir+'Cepheids-data.csv') # loading the Cepheids data
hubble = pd.read_csv(datadir+'Hubble-data.csv')  # loading the Hubble data

# Defining the fit model functions
def cepheids_model(x, a, b):  # cepheids fit model
    return a*np.log10(x)+b


def hubbles_model(x, H):  # hubbles fit model
    return H*x

# fitting data on cepheids model
parameters, covariance = curve_fit(
    cepheids_model, cepheids['P (day)'], cepheids['M'], sigma=1/(cepheids['std_M'])**2)
C_error = np.sqrt(linalg.eigvals(covariance))

# fitting data on hubbles model
H, H_covariance = curve_fit(
    hubbles_model, hubble['d (Mpc)'], hubble['V (km/s)'], sigma=hubble['std_V'])
H_error = np.sqrt(H_covariance[0])

# Plotting the results

plt.figure()
plt.plot(np.sort(cepheids['P (day)'].values), cepheids_model(
    np.sort(cepheids['P (day)'].values), parameters[0], parameters[1]), label='Fitted model, $M_v=a \\times log(P)+b$\n' + f'a={round(parameters[0], 5)}$\pm {round(C_error.real[0], 5)}$, b={round(parameters[1], 5)}$\pm{round(C_error.real[1], 5)}$', color='black')
plt.errorbar(cepheids['P (day)'].values, cepheids['M'].values, yerr=cepheids['std_M'], label='Data',
             color='darkred', fmt='+', capsize=2)
plt.title('The Log(P)-M relation for cepheids')
plt.ylabel('Absolute Magnitude')
plt.xlabel('$Log(P)$')
plt.xscale('log')
plt.legend()
plt.savefig(figuredir+'cepheid.png')

plt.figure()
plt.plot(np.sort(hubble['d (Mpc)'].values), hubbles_model(
    np.sort(hubble['d (Mpc)'].values), H[0]), label=f'Fitted model, V=H.r\nH={round(H[0], 5)}$\pm{round(H_error[0], 5)}$', color='black')
plt.errorbar(hubble['d (Mpc)'], hubble['V (km/s)'], xerr=hubble['std_d'], yerr=hubble['std_V'],
             label='Data', color='darkred', fmt='+', capsize=2)
plt.title("Hubble's Equation with considering y-data errors")
plt.xlabel('d(Mpc)')
plt.ylabel('V(km/s)')
plt.legend()
plt.savefig(figuredir+'hubble.png')
