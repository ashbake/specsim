import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import pathlib
import pandas as pd


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
    """Takes a string and removes the first underscore and everything to the left of it."""
    # Find the position of the first underscore
    first_underscore_index = string.find('_')
    
    # Return the substring from the first underscore to the end. If no underscore is found, return the original string
    if first_underscore_index != -1:
        return string[first_underscore_index + 1:]
    return string


def error_handled_fits(file_path):
    """Process a single FITS file and return the radius and profiles, with extra error handling in case the file is empty."""
    try:
        # Check if the file is empty (size is 0)
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File {file_path} is empty.")
        else:
            # Attempt to open the file with fits.open() if it's not empty
            with fits.open(file_path) as hdul:  
                profiles = {}
                bin_centers = {}
                dp = {}
                radii = {}
                for i in range(0, len(hdul)):
                    dp [f'extension_{i}'] = hdul[i].header['DP']
                    data = hdul[i].data
                    center = (data.shape[1] // 2, data.shape[0] // 2)
                    profiles[f'extension_{i}'], bin_centers[f'extension_{i}'] = radial_profile(data, center)
                    radii[f'extension_{i}'] = bin_centers[f'extension_{i}'] * dp[f'extension_{i}'] # convert pixels to angular separation in arcsec
            return radii, profiles
        
    except ValueError as e:
        # Handle the case where the file is empty
        print(e)
    except Exception as e:
        # Handle other exceptions, such as file not found
        print(f"An error occurred: {e}")


def convert_path_to_name(file_path_str):
    """Converts a file path to a name string based on the folder structure."""
    file_path = pathlib.Path(file_path_str)
    parent_path = file_path.parent
    subfolders_count = len([part for part in parent_path.parts if part != parent_path.drive])
    names = []
    # new_name = ""
    for i in range (0, subfolders_count+1):
        names.append(file_path.parts[i])

    new_name = "_".join(names)
    new_name = remove_left_of_first_underscore(new_name)
    
    if new_name.endswith('.fits'):
        return new_name[:-5]  # Remove the last 5 characters
    
    return new_name


def convert_files_to_csv(root_directory, output_directory):
    """Goes through a directory and converts FITS files to csv files containing the radii and profiles in a new folder."""
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for file in filenames:
            # Check if the file is a FITS file and matches the desired pattern
            if file.endswith('_x0_y0.fits'):
                # Create the full path to the file
                full_path = os.path.join(dirpath, file)

                # Name the new file based on the folders it's in
                csv_name_base = convert_path_to_name(full_path)

                result = error_handled_fits(full_path)
                if result is not None:
                    # Get radii and profiles using the error_handled_fits function
                    radii, profiles = error_handled_fits(full_path)

                    # Create a DataFrame for each extension
                    for ext in radii.keys():
                        df = pd.DataFrame({
                            'Radius (arcsec)': radii[ext],
                            'Profile (intensity)': profiles[ext]
                        })

                        # Use the label mapping for the extension name
                        label = label_mapping.get(ext, ext)

                        # Create the new CSV filename with underscores
                        new_filename = f"{csv_name_base}_{label}.csv"

                        # Save the DataFrame to a CSV file in the output directory
                        csv_path = os.path.join(output_directory, new_filename)
                        df.to_csv(csv_path, index=False)

                        print(f"Converted {full_path} {ext} to {csv_path}")
                else: 
                    print(f"Error processing {full_path}")


def plot_from_csv(csv_file):
    """Plot the csv data to show radial profile."""
    # Read the CSV file, skipping the header row
    df = pd.read_csv(csv_file)

    # Assuming the CSV has two columns, use their names to access the data
    x = df.iloc[:, 0]  # First column
    y = df.iloc[:, 1]  # Second column
    band = csv_file.split('_')[-1][:-4]
    csv_name = os.path.basename(csv_file)

    plt.figure(figsize=(10, 6))
    plt.loglog(x, y, label = band)
    plt.xlabel('Radius (arcseconds)', fontweight='bold', fontsize=14)
    plt.ylabel('Intensity', fontweight='bold', fontsize=14)
    plt.title(f'Radial Profile for {csv_name}', fontweight='bold', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
