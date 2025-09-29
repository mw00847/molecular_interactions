
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
from sklearn.metrics import mean_squared_error

#voigt function definition
def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile: Combination of Gaussian (sigma) and Lorentzian line shapes."""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

#function to fit voigt profile to a spectrum
def fit_voigt(x, y, initial_center, initial_sigma=10, initial_gamma=10):
    """Fits a Voigt function to the given data."""
    p0 = [max(y), initial_center, initial_sigma, initial_gamma]
    bounds = ([0, initial_center - 20, 0.1, 0.1], [np.inf, initial_center + 20, 50, 50])
    popt, pcov = curve_fit(voigt, x, y, p0=p0, bounds=bounds)
    return popt, pcov

#baseline fitting function using polynomial regression
def fit_baseline(wavenumbers, intensities, degree=3, baseline_regions=None):
    """Fits a polynomial baseline and subtracts it."""
    if baseline_regions is None:
        baseline_regions = [(1660, 1670), (1730, 1740)]

    #select baseline points
    mask = np.zeros_like(wavenumbers, dtype=bool)
    for region in baseline_regions:
        mask |= (wavenumbers >= region[0]) & (wavenumbers <= region[1])

    baseline_x = wavenumbers[mask]
    baseline_y = intensities[mask]

    #fit polynomial to baseline regions
    poly_coeffs = np.polyfit(baseline_x, baseline_y, deg=degree)
    baseline = np.polyval(poly_coeffs, wavenumbers)

    #subtract baseline
    corrected_intensities = intensities - baseline

    return corrected_intensities, baseline

#compute RMSE
def calculate_rmse(y_exp, y_fit):
    return np.sqrt(mean_squared_error(y_exp, y_fit))

#compute R² Score
def calculate_r2(y_exp, y_fit):
    ss_res = np.sum((y_exp - y_fit) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return 1 - (ss_res / ss_tot)

#main processing function
def process_and_store_results(file_names, peak_center_guess):
    """Fits Voigt profiles, plots results, and stores them in a NumPy array."""
    results = []
    
    #initialize lists for NumPy array
    file_names_list = []
    amplitudes = []
    centers = []
    sigmas = []
    gammas = []

    for file_name in file_names:
        # Load the spectrum
        data = pd.read_csv(file_name, names=['Wavenumbers', 'Intensities'])
        wavenumbers = data['Wavenumbers'].values
        intensities = data['Intensities'].values

        #fit and subtract baseline
        corrected_intensities, baseline = fit_baseline(wavenumbers, intensities, degree=3)

        #select a region around the peak
        region_mask = (wavenumbers > peak_center_guess - 50) & (wavenumbers < peak_center_guess + 50)
        region_wavenumbers = wavenumbers[region_mask]
        region_intensities = corrected_intensities[region_mask]  # Use baseline-corrected intensities

        #fit the Voigt function
        popt, pcov = fit_voigt(region_wavenumbers, region_intensities, peak_center_guess)

        #store in list for NumPy array
        file_names_list.append(file_name)
        amplitudes.append(popt[0])
        centers.append(popt[1])
        sigmas.append(popt[2])
        gammas.append(popt[3])

        #store as dictionary for reference
        results.append({
            'File': file_name,
            'Amplitude': popt[0],
            'Center': popt[1],
            'Sigma (Gaussian width)': popt[2],
            'Gamma (Lorentzian width)': popt[3],
            'Fit Parameters': popt
        })

        #generate fitted curve and residuals
        fit_curve = voigt(region_wavenumbers, *popt)
        residuals = region_intensities - fit_curve

        #plot original spectrum and Voigt fit
        plt.figure(figsize=(10, 6))
        plt.plot(region_wavenumbers, region_intensities, label='Baseline Corrected Spectrum', color='blue')
        plt.plot(region_wavenumbers, fit_curve, linestyle='--', label='Voigt Fit', color='red')
        plt.title(f"Voigt Fit for {file_name.split('/')[-1]}")
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.legend()        
        plt.xlim(popt[1] + 20, popt[1] - 20)
        plt.ylim(0, None)
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #plot residuals
        plt.figure(figsize=(10, 4))
        plt.plot(region_wavenumbers, residuals, label='Residuals', color='purple')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Residuals for {file_name.split('/')[-1]}")
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Residual Intensity')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #convert to structured NumPy array
    results_array = np.column_stack((amplitudes, centers, sigmas, gammas))

    return results, results_array

#list of file names and peak center guess
file_names = [
    '0.CSV',
    '1.CSV',
    '2.CSV',
    '3.CSV',
    '4.CSV',
    '5.CSV',
    '6.CSV',
    '7.CSV',
    '8.CSV',
    '9.CSV',
    '10.CSV',
]

peak_center_guess = 1709  # Adjust based on expected peak location

#process spectra and create plots
fit_results, fit_array = process_and_store_results(file_names, peak_center_guess)

#display fit results
for result in fit_results:
    print(f"File: {result['File']}")
    print(f"  Amplitude: {result['Amplitude']:.4f}")
    print(f"  Center: {result['Center']:.4f} cm⁻¹")
    print(f"  Sigma (Gaussian width): {result['Sigma (Gaussian width)']:.4f}")
    print(f"  Gamma (Lorentzian width): {result['Gamma (Lorentzian width)']:.4f}")
    print()

#check the NumPy array
print("NumPy Array of Fitting Parameters:")
print(fit_array)
