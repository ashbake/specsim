import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import pathlib
import csv
import re

#First 4 functions are from Dimitri

def radial_profile(data, center):
    """ Calculate the radial profile of a 2D array. """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Define the desired thickness
    thickness = 4  # thickness in pixels
    
    # Bin the pixel values by their distances
    r_flat = r.ravel()
    image_flat = data.ravel()
    # Define bin edges with the desired thickness
    bin_edges = np.arange(0, r.max() + thickness, thickness)

    # Calculate the mean intensity within each bin
    radial_profile, bin_edges, binnumber = binned_statistic(r_flat, image_flat, statistic='mean', bins=bin_edges)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return radial_profile, bin_centers


def process_fits_file(file_path):
    """ Process a single FITS file and return the radial profiles. """
    with fits.open(file_path) as hdul:  
        profiles = {}
        bin_centers = {}
        dp = {} # Dictionary to store the pixel scale for each extension
        for i in range(0, len(hdul)):
            dp [f'extension_{i}'] = hdul[i].header['DP']
            data = hdul[i].data
            center = (data.shape[1] // 2, data.shape[0] // 2)
            profiles[f'extension_{i}'], bin_centers[f'extension_{i}'] = radial_profile(data, center)
    return profiles, bin_centers, dp


def process_directory(directory):
    """ Process all FITS files in the given directory and subdirectories. """
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.fits'):
                file_path = os.path.join(root, file)
                profiles, bin_centers, dp = process_fits_file(file_path)
                rel_dir = os.path.relpath(root, directory)
                if rel_dir not in results:
                    results[rel_dir] = []
                results[rel_dir].append((file, profiles, bin_centers, dp))
    return results


label_mapping = {
    'extension_0': 'y',
    'extension_1': 'J',
    'extension_2': 'H',
    'extension_3': 'K',
}


def plot_radial_profiles(results):
    """ Plot the radial profiles from the results dictionary. """
    for rel_dir, files_data in results.items():
        for file, profiles, bin_centers, dp in files_data:
            plt.figure(figsize=(10, 6))
            for ext, profile in profiles.items():
                radii = bin_centers[ext] * dp[ext]  # Convert pixel radius to arcseconds
                label = label_mapping.get(ext, ext)  # Default to ext if no mapping is found
                plt.loglog(radii, profile, label=label)
            plt.xlabel('Radius (arcseconds)', fontweight='bold')
            plt.ylabel('Intensity', fontweight='bold')
            plt.title(f'Radial Profile for {file}', fontweight='bold')
            plt.legend()
            plt.grid(True)
            plt.show()

def remove_left_of_first_underscore(string):
    # Find the position of the first underscore
    first_underscore_index = string.find('_')
    
    # Return the substring from the first underscore to the end
    # If no underscore is found, return the original string
    if first_underscore_index != -1:
        return string[first_underscore_index + 1:]
    return string

def convert_path_to_name(file_path_str):
    file_path = pathlib.Path(file_path_str)
    parent_path = file_path.parent
    subfolders_count = len([part for part in parent_path.parts if part != parent_path.drive])
    names = []
    new_name = ""
    for i in range (0, subfolders_count+1):
        names.append(test_path.parts[i])

    new_name = "_".join(names)
    new_name = remove_left_of_first_underscore(new_name)
    
    return new_name
