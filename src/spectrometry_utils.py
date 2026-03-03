import os
import subprocess
import pandas as pd
import numpy as np
from pyopenms import *
from scipy.signal import find_peaks
from tqdm import tqdm  # Importer tqdm pour la barre de progression

def convert_raw_to_mzml(raw_files_path, output_dir):
    command = [
        "msconvert", f"{raw_files_path}/*.RAW",
        "--filter", "lockmassRefiner mz=524 tol=1.0",
        "-o", output_dir, "--mzML"
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

def load_data(directories, class_names):
    all_data = pd.DataFrame()
    for directory, class_name in zip(directories, class_names):
        file_list = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".mzML")]

        for filename in file_list:
            exp = MSExperiment()
            MzMLFile().load(filename, exp)

            chromatogram = exp.getChromatograms()[0]
            times, intensities = chromatogram.get_peaks()

            summ = sum(intensities)
            h = 0.04 * summ
            peaks, _ = find_peaks(intensities, height=h)

            spectra = exp.getSpectra()
            pic_spectra = [spectra[peak_index] for peak_index in peaks]

            all_mz_values = set()
            for spectrum in pic_spectra:
                mz_values, _ = spectrum.get_peaks()
                all_mz_values.update(mz_values)

            all_mz_values = sorted(all_mz_values)
            rts = []
            sums = []
            data = []

            for spectrum in pic_spectra:
                mz_values, intensities = spectrum.get_peaks()
                rt_value = spectrum.getRT()
                rts.append(rt_value)
                intensity_dict = dict(zip(mz_values, intensities))
                row_data = {mz: intensity_dict.get(mz, np.nan) for mz in all_mz_values}
                row_data['Class'] = class_name
                row_data['File'] = os.path.basename(filename)
                row_data['RT'] = rt_value
                row_data['Sum'] = sum(intensities)
                data.append(row_data)
            file_data = pd.DataFrame(data)
            all_data = pd.concat([all_data, file_data], ignore_index=True)

    desired_columns = ['Class', 'File', 'RT', 'Sum'] + sorted(all_mz_values)
    all_data = all_data.reindex(columns=desired_columns)
    return all_data

def preprocess_data(all_data):
    df1 = all_data.iloc[:, :4]
    df2 = all_data.T.iloc[4:]
    df2 = df2.fillna(df2.min(axis=0), axis=0)
    df2 = df2.T
    dfs = [df1['Class'], df1['File'], df1['RT'], df1['Sum'], df2]
    df_concatenated = pd.concat(dfs, axis=1)
    df_concatenated.iloc[:, 4:] = df_concatenated.iloc[:, 4:].div(df_concatenated['Sum'], axis=0)
    return df_concatenated

def savecsv(data, name="mydata.csv"):
    data.to_csv(name, index=False)

def readcsv(path):
    data = pd.read_csv(path)
    return data
