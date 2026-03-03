import os
import time
import subprocess
import pandas as pd
import numpy as np
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pyopenms import MSExperiment, MzMLFile, PeakMap
from scipy.signal import find_peaks
import sklearn
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
from scipy.stats import mode

models = []
numeric_features = []

# Define the path for the stop signal file
stop_signal_file = 'stop_signal.txt'

def load_models(model_paths):
    global models
    models = [joblib.load(path) for path in model_paths]
    print("Models have been loaded.")

def load_numeric_features(nf_path):
    global numeric_features
    numeric_features = joblib.load(nf_path)
    print("Numeric features have been loaded.")

def convert_raw_to_mzml(raw_file_path, output_dir):
    command = [
        "msconvert", raw_file_path,
        "--filter", "lockmassRefiner mz=524 tol=1.0",
        "-o", output_dir, "--mzML"
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

def load_data(file_path):
    exp = MSExperiment()
    MzMLFile().load(file_path, exp)

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
        row_data['File'] = os.path.basename(file_path)
        row_data['RT'] = rt_value
        row_data['Sum'] = sum(intensities)
        data.append(row_data)
    
    all_data = pd.DataFrame(data)
    desired_columns = ['File', 'RT', 'Sum'] + sorted(all_mz_values)
    all_data = all_data.reindex(columns=desired_columns)
    return all_data

def preprocess_data(all_data):
    df1 = all_data.iloc[:, :3]
    df2 = all_data.T.iloc[3:]
    df2 = df2.fillna(df2.min(axis=0), axis=0)
    df2 = df2.T
    dfs = [df1['File'], df1['RT'], df1['Sum'], df2]
    df_concatenated = pd.concat(dfs, axis=1)
    df_concatenated.iloc[:, 3:] = df_concatenated.iloc[:, 3:].div(df_concatenated['Sum'], axis=0)
    return df_concatenated

def decision(processed_data):
    processed_data = processed_data.drop(['File', 'RT', 'Sum'], axis=1, errors='ignore')
    processed_data = processed_data.reindex(columns=numeric_features, fill_value=np.float64(0))
    
    predictions = []
    for model in models:
        predictions.append(model.predict(processed_data))
    
    predictions_df = pd.DataFrame(predictions).T
    majority_predictions = mode(predictions_df, axis=1)[0].flatten()
    
    return majority_predictions

def is_file_fully_created(file_path, check_interval=1, stable_duration=5):
    stable_start_time = None
    last_size = os.path.getsize(file_path)

    while True:
        time.sleep(check_interval)
        current_size = os.path.getsize(file_path)

        if current_size != last_size:
            stable_start_time = None
            last_size = current_size
        else:
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= stable_duration:
                return True

class Watcher:
    def __init__(self, watch_directory, output_directory):
        self.watch_directory = watch_directory
        self.output_directory = output_directory
        self.observer = Observer()
        self.processed_data = None

    def run(self):
        event_handler = Handler(self.output_directory, self)
        self.observer.schedule(event_handler, self.watch_directory, recursive=False)
        self.observer.start()
        print(f"Watching directory: {self.watch_directory}")
        try:
            while not os.path.exists(stop_signal_file):
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.stop()
        self.observer.join()

    def update_processed_data(self, data):
        self.processed_data = data

class Handler(FileSystemEventHandler):
    def __init__(self, output_directory, watcher):
        self.output_directory = output_directory
        self.watcher = watcher

    def on_created(self, event):
        print(f"Created event detected: {event.src_path}")
        raw_file_path = event.src_path
        print(f"New file detected: {raw_file_path}")

        if not is_file_fully_created(raw_file_path):
            print(f"File {raw_file_path} is not fully created. Skipping processing.")
            return

        print("Converting RAW to mzML...")
        try:
            convert_raw_to_mzml(raw_file_path, self.output_directory)
            print("Conversion complete")

            mzml_file_path = os.path.join(self.output_directory, os.path.splitext(os.path.basename(raw_file_path))[0] + '.mzML')
            data = load_data(mzml_file_path)
            print("Data loaded")

            processed_data = preprocess_data(data)
            print("Data preprocessed")

            diagnosis = decision(processed_data)
            print("Predictions:", diagnosis)

        except Exception as e:
            print(f"Error processing file: {raw_file_path}")
            print(e)

def start_watcher(watch_directory, output_directory):
    global watcher
    watcher = Watcher(watch_directory, output_directory)
    watcher.run()

def stop_watcher():
    with open(stop_signal_file, 'w') as f:
        f.write('Stop')

def load_models_and_features(model_paths, nf_path):
    load_models(model_paths)
    load_numeric_features(nf_path)
