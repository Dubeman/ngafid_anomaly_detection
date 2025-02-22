import json
import os
import logging
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class TestResult:
    total_files: int
    directory_file_count: int
    is_equal: bool
    missing_files: list

def calculate_total_files(json_file_path, directory_files):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    total_files = 0
    missing_files = []

    for event in data:
        before_flights = [f for f in event.get('before_flights', []) if f in directory_files]
        day_of_flights = [f for f in event.get('day_of_flights', []) if f in directory_files]
        after_flights = [f for f in event.get('after_flights', []) if f in directory_files]

        total_files += len(before_flights)
        total_files += len(day_of_flights)
        total_files += len(after_flights)

        # Log missing files
        missing_files.extend([f for f in event.get('before_flights', []) if f not in directory_files])
        missing_files.extend([f for f in event.get('day_of_flights', []) if f not in directory_files])
        missing_files.extend([f for f in event.get('after_flights', []) if f not in directory_files])

    return total_files, missing_files

def count_files_in_directory(directory_path):
    return [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name)) and name.endswith('.csv')]

def perform_file_check(json_file_path, directory_path):
    directory_files = count_files_in_directory(directory_path)
    total_files, missing_files = calculate_total_files(json_file_path, directory_files)
    directory_file_count = len(directory_files)
    
    is_equal = total_files == directory_file_count
    
    return TestResult(total_files, directory_file_count, is_equal, missing_files)

def check_files_open(json_file_path, directory_path):
    directory_files = count_files_in_directory(directory_path)
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    non_openable_files = []

    for event in data:
        all_files = event.get('before_flights', []) + event.get('day_of_flights', []) + event.get('after_flights', [])
        for file_name in all_files:
            if file_name in directory_files:
                try:
                    with open(os.path.join(directory_path, file_name), 'r') as f:
                        pass
                except Exception as e:
                    non_openable_files.append(file_name)
    
    return non_openable_files



def plot_altitude_agl(directory_path, directory_files):
    for file_name in directory_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            df = pd.read_csv(file_path)
            if 'AltitudeAGL' in df.columns:
                plt.figure()
                plt.plot(df.index, df['AltitudeAGL'])
                plt.title(f'AltitudeAGL vs Index for {file_name}')
                plt.xlabel('Index')
                plt.ylabel('AltitudeAGL')
                plt.show()
            else:
                logging.warning(f"'AltitudeAGL' column not found in {file_name}")
        except Exception as e:
            logging.error(f"Error reading {file_name}: {e}")





def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_test_result(result, json_file_path):
    logging.info(f"Testing JSON file: {json_file_path}")
    logging.info(f"Total number of files: {result.total_files}")
    logging.info(f"Number of files in directory: {result.directory_file_count}")
    
    if result.is_equal:
        logging.info("The sum of the total files is equal to the number of files in the directory.")
    else:
        logging.warning("The sum of the total files is not equal to the number of files in the directory.")
        logging.warning(f"Missing files: {result.missing_files}")

def log_non_openable_files(non_openable_files, json_file_path):
    if non_openable_files:
        logging.warning(f"Files in {json_file_path} that could not be opened: {non_openable_files}")
    else:
        logging.info(f"All files in {json_file_path} could be opened successfully.")

def main():
    setup_logging()
    
    json_file_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/metadata/c37_cluster_events.json'
    directory_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/data/c37_cleaned'
    
    result = perform_file_check(json_file_path, directory_path)
    log_test_result(result, json_file_path)
    
    non_openable_files = check_files_open(json_file_path, directory_path)
    log_non_openable_files(non_openable_files, json_file_path)
    
    # Check with the altered JSON file
    altered_json_file_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/metadata/c37_cluster_events_altered.json'
    
    altered_result = perform_file_check(altered_json_file_path, directory_path)
    log_test_result(altered_result, altered_json_file_path)
    
    non_openable_files_altered = check_files_open(altered_json_file_path, directory_path)
    log_non_openable_files(non_openable_files_altered, altered_json_file_path)
    
    directory_files = count_files_in_directory(directory_path)
    plot_altitude_agl(directory_path, directory_files)

if __name__ == "__main__":
    main()