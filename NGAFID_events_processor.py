import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
import logging
import imageio
import csv
import json
import math
from utils import extract_keys, get_last_processed_file, save_last_processed_file, save_parameters_to_file
import uuid


# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COLUMNS_TO_DROP = config['COLUMNS_TO_DROP']
MIN_TIME_STEPS_PER_FILE = config['MIN_TIME_STEPS_PER_FILE']
CUTOFF = config['CUTOFF']
TIME_THRESHOLD = config['TIME_THRESHOLD']
LOAD_FOLDER_PATH = config['LOAD_FOLDER_PATH'] 
SAVE_FRAMES_AS_GIF = config['SAVE_FRAMES_AS_GIF']
LAST_PROCESSED_FILE_PATH = config['LAST_PROCESSED_FILE_PATH']
PARAMS_SAVE_FILE_PATH = config['PARAMS_SAVE_FILE_PATH']
FRAMES_DIR = config['FRAMES_DIR']
EVENT_DEFINITION = config['EVENT_DEFINITION']
GIF_SAVE_FOLDER_PATH = config['GIF_SAVE_FOLDER_PATH']

#possibly deperecated globals
COMBINED_DATA_PATH = config['COMBINED_DATA_PATH']
FILES_WITH_VALID_ALTMSL_PATH = config['FILES_WITH_VALID_ALTMSL_PATH']

#


frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)  # Ensure frames directory exists
frame_count = 0  # Track frames
class NGAFIDEventsPreprocessor:
    '''
    Preprocess the day of events data from the NGAFID. For any change in the pipeline,
    the class should be updated accordingly. the ipynb can be used as the testing ground.
    Also has CLI and GUI options for the user to interact with the data. TODO : separate the CLI and GUI options into separate classes
    '''
    def __init__(self, load_folder_path=None, save_folder_path=None):
        self.folder_path = load_folder_path
        print(f"Loading data from {load_folder_path}...")
        self.files_with_nan_altmsl = []
        self.file_paths = None # List of file paths
        self.save_folder = save_folder_path
        self.num_sections_saved = 0
        self.EVENT_STATE = None

        # if save folder does not exist, create it
        if save_folder_path and not os.path.exists(save_folder_path):
            print(f"Creating save folder at {save_folder_path}...")
            os.makedirs(save_folder_path)



    def get_file_paths(self, combine_files=True):
        """Get the file paths based on the existence of combined_data.csv."""
        # Find all CSV files matching the pattern *_day_of_*.csv
        csv_files = glob.glob(os.path.join(self.folder_path, '*_day_of_*.csv'))

        # Check if combined_data.csv exists
        combined_data_path = os.path.join(self.folder_path, COMBINED_DATA_PATH)
        if not os.path.exists(combined_data_path) and combine_files:
            # If combined_data.csv does not exist, combine CSV files and create it
            self.combine_csv_files(csv_files, combined_data_path)
            self.create_file_names_txt(csv_files)
            self.file_paths = csv_files
            return [combined_data_path]
        else:
            # If combined_data.csv exists, use it as the only file path
            self.file_paths = [combined_data_path]
            return [combined_data_path]

    def combine_csv_files(self, csv_files, combined_data_path, N_FILES=None):
        """Combine multiple CSV files into a single DataFrame and save it to a new CSV file."""
        combined_df = pd.DataFrame()

        # Loop through each file and concatenate them into one DataFrame
        for idx, file in enumerate(csv_files[:N_FILES]):
            df = pd.read_csv(file, skiprows=2)
            # Add a unique identifier to each row based on the file index
            df['id'] = f'file_{idx}'
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        combined_df.to_csv(combined_data_path, index=False)

    def create_file_names_txt(self, csv_files):
        """Create a file_names.txt file with the list of CSV files."""
        file_names_path = os.path.join(self.folder_path, 'file_names.txt')
        with open(file_names_path, 'w') as f:
            for file in csv_files:
                f.write(f"{file}\n")

    def load_data(self, file_path, skiprows=2, columns_to_drop=None):
        """Load the CSV file into a pandas DataFrame, skipping the first two rows."""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, skiprows=skiprows, low_memory=False)
        df.columns = df.columns.str.strip()  # Strip leading/trailing whitespace from column names
        
        # Drop columns if specified
        if columns_to_drop:
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        self.current_df = df
        return df

    def convert_columns(self, df):
        """Convert relevant columns to appropriate data types."""
        df['AltMSL'] = pd.to_numeric(df['AltMSL'], errors='coerce')  # Convert 'AltMSL' to numeric
        df['Lcl Time'] = pd.to_datetime(df['Lcl Time'].str.strip(), format='%H:%M:%S', errors='coerce')  # Convert 'Lcl Time' to datetime
        return df

    def drop_nan_altmsl(self, df, file_path):
        """Drop rows where 'AltMSL' is completely NaN."""
        if df['AltMSL'].isna().all():
            self.files_with_nan_altmsl.append(file_path)
            return df.dropna(subset=['AltMSL'])
        return df
    

    def filter_files_with_valid_altmsl(self):
        """Filter out files with completely NaN 'AltMSL' from file paths and save the valid ones."""
        valid_files_path = os.path.join(self.folder_path, FILES_WITH_VALID_ALTMSL_PATH)
        valid_files = [fp for fp in self.file_paths if fp not in self.files_with_nan_altmsl] # Filter out files with complete NaN 'AltMSL'
        
        valid_files_df = pd.DataFrame(valid_files, columns=['Filename'])
        valid_files_df.to_csv(valid_files_path, index=False)
        
        self.file_paths = valid_files

    def preprocess_file(self, file_path, columns_to_drop=None):
        """Preprocess a single file."""
        df = self.load_data(file_path)
        df = self.convert_columns(df)
        df = self.drop_nan_altmsl(df, file_path)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')

        if self.save_folder:
            file_name = os.path.basename(file_path)
            save_path = os.path.join(self.save_folder, file_name)
            #print saving {file name} to the save path
            print(f"Saving {file_path} to {save_path}...")


            df.to_csv(save_path, index=False)
        return df

    def preprocess_all_files(self, columns_to_drop=None):
        """Preprocess all files and return a list of DataFrames."""
        # Set the file paths to the list of files in the folder
        self.file_paths = glob.glob(os.path.join(self.folder_path, '*'))

        dataframes = []
        
        for file_path in self.file_paths:
            print("---------------------------------")
            if file_path.endswith('.csv'):
                print(f"Preprocessing {os.path.basename(file_path)}...")
                df = self.preprocess_file(file_path, columns_to_drop)
                dataframes.append(df)
            elif file_path.endswith('.txt'):
                save_path = os.path.join(self.save_folder, os.path.basename(file_path))
                print(f"Copying {os.path.basename(file_path)} to {save_path}...")
                with open(file_path, 'r') as f_src:
                    with open(save_path, 'w') as f_dst:
                        f_dst.write(f_src.read())
        return dataframes

    def get_files_with_nan_altmsl(self):
        """Return the list of files with completely NaN 'AltMSL' column."""
        return self.files_with_nan_altmsl

    def describe_column(self, df, column_name):
        """Provide summary statistics for a specified column."""
        return df[column_name].describe()
    
    def process_cutoff_AGL(self, df, filepath=None, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD, min_time_steps=MIN_TIME_STEPS_PER_FILE):
        """Process the AltitudeAGL column to identify sections where the altitude crosses a specified cutoff value.
        The function splits the flight data into segments based on altitude and time thresholds, ensuring each segment
        meets the minimum required time steps. The processed segments are then saved to separate CSV files with appropriate
        naming conventions based on the event state (before, day_of, after). Additionally, the function logs the processing
        details and updates the last processed file and parameters."""

        df['AltitudeAGL'] = pd.to_numeric(df['AltitudeAGL'], errors='coerce')
        df = df.dropna(subset=['AltitudeAGL'])
        original_file_name = os.path.splitext(os.path.basename(filepath))[0]
        crossing_indices = self.get_crossing_indices(df, cutoff=cutoff, time_threshold=time_threshold)

        original_file_name = os.path.splitext(os.path.basename(filepath))[0]

        df_list = []
        # Store the indices where the AltitudeAGL crosses the cutoff
        crossing_indices = self.get_crossing_indices(df, cutoff=cutoff, time_threshold=time_threshold)

        # Ensure we sort the crossing indices in ascending order
        crossing_indices = sorted(crossing_indices)
        logging.info(f'Using crossing indices: {crossing_indices}')

        # Save the sections to a new csv file with _section_{n} appended to the original file name
        for i in range(len(crossing_indices) - 1):
            start_index = int(crossing_indices[i])
            end_index = int(crossing_indices[i + 1])
            logging.info(f"Processing section from index {start_index} to {end_index}...")
            section_df = df.iloc[start_index:end_index]
            
            if len(section_df) > min_time_steps:
                df_list.append((section_df, start_index, end_index))
            else:
                logging.info(f"Skipping section from index {start_index} to {end_index} due to insufficient time steps...")
                continue

        logging.info(f"Found {len(df_list)} sections in {original_file_name}...")


        section_counter = 0
        for section_df, _, _ in df_list:
            match_1 = re.search(r'_before_(\d+)', original_file_name)
            match_2 = re.search(r'_day_of_(\d+)', original_file_name)
            match_3 = re.search(r'_after_(\d+)', original_file_name)

            if match_1:
                before_number = int(match_1.group(1))
                file_name = f"event_{self.EVENT_STATE['event_id']}_before_{before_number}_section_{section_counter}"
                section_file_path = f"{self.save_folder}/{file_name}.csv"
                section_df.to_csv(section_file_path, index=False)

            elif match_2:
                day_of_number = str(match_2.group(1))[-2:]
                file_name = f"event_{self.EVENT_STATE['event_id']}_day_of_{day_of_number}_section_{section_counter}"
                section_file_path = f"{self.save_folder}/{file_name}.csv"
                section_df.to_csv(section_file_path, index=False)

            elif match_3:
                after_number = int(match_3.group(1))
                file_name = f"event_{self.EVENT_STATE['event_id']}_after_{after_number}_section_{section_counter}"
                section_file_path = f"{self.save_folder}/{file_name}.csv"
                section_df.to_csv(section_file_path, index=False)

            else:
                print("No match found...")
                section_file_path = None

            if section_file_path:
                logging.info(f"\033[95mSaved section {section_counter} to {section_file_path}...\033[0m")

            original_file_path = os.path.join(self.folder_path, original_file_name)
            save_last_processed_file(original_file_path, LAST_PROCESSED_FILE_PATH)
            logging.info(f"\033[94mUpdated last processed file to {original_file_path}...\033[0m")
            section_counter += 1

        # Save the parameters (cutoff, time_threshold, min_time_steps) to a file
        params = {'Cutoff': cutoff, 'Time Threshold': time_threshold, 'Min Time Steps': min_time_steps}
        save_parameters_to_file(params, PARAMS_SAVE_FILE_PATH)
        logging.info(f"\033[94mSaved parameters to {cutoff}, {time_threshold}, {min_time_steps} to {PARAMS_SAVE_FILE_PATH}...\033[0m")
        self.num_sections_saved = len(df_list)  
        return self.num_sections_saved
    
    # def get_crossing_indices(self, df, cutoff=50, time_threshold=TIME_THRESHOLD):
    #     """Get the crossing indices where the AltitudeAGL crosses the cutoff."""
    #     crossing_indices = []
    #     start_index = None
    #     end_index = None
    #     below_cutoff_start = None
    #     above_cutoff_start = None


    #     for i in range(1,len(df)-1):
                
    #             if start_index is not None and end_index is not None:
    #                     crossing_index = (start_index + end_index) // 2
    #                     crossing_indices.append(crossing_index)
    #                                             # Reset tracking variables for the next segment
    #                     start_index = None
    #                     end_index = None
    #                     below_cutoff_start = None
    #                     above_cutoff_start = None
 

    #             if math.isclose(df['AltitudeAGL'].iloc[i], cutoff, abs_tol=0):  # Descent

    #                 if (df['AltitudeAGL'].iloc[i+1] - df['AltitudeAGL'].iloc[i] < 0) and (df['AltitudeAGL'].iloc[i] - df['AltitudeAGL'].iloc[i-1] < 0) :
    #                     print("found a descent at index and height :", i, df['AltitudeAGL'].iloc[i])    
    #                     if below_cutoff_start is None:
    #                         below_cutoff_start = i
    #                     if below_cutoff_start is None:
    #                         below_cutoff_start = i  # Start potential descent tracking

    #                     if i - below_cutoff_start > time_threshold and start_index is None:
    #                         start_index = below_cutoff_start  # Confirm descent event
    #                         below_cutoff_start = None  # Reset timer



    #                 if (df['AltitudeAGL'].iloc[i+1] - df['AltitudeAGL'].iloc[i] > 0) and (df['AltitudeAGL'].iloc[i] - df['AltitudeAGL'].iloc[i-1] > 0):
    #                     print("found an ascent at index", i)    
    #                     if above_cutoff_start is None:
    #                         above_cutoff_start = i
    #                     if above_cutoff_start is not None:
    #                         above_cutoff_start = i
    #                     if i - above_cutoff_start > time_threshold and end_index is None:
    #                         end_index = above_cutoff_start
    #                         above_cutoff_start = None

   
    def get_crossing_indices(self, df, cutoff=80, time_threshold=TIME_THRESHOLD):
        """Get the crossing indices where the AltitudeAGL crosses the cutoff."""
        crossing_indices = []
        gradient_indices = []
        grad_diff = 0

        for i in range(grad_diff,len(df)-grad_diff):
                
                if math.isclose(df['AltitudeAGL'].iloc[i], cutoff, rel_tol=1e-1):  # Descent

                    gradient_indices.append(i)

        if gradient_indices:
            gradient_indices.insert(0, 0)  # Insert the first index
            gradient_indices.append(len(df) - 1)  # Append the last index
        i = 0
        while i < len(gradient_indices) - 1:
            if gradient_indices[i + 1] - gradient_indices[i] > time_threshold:
                # also check if most of the values in the range are below the cutoff
                if df['AltitudeAGL'].iloc[gradient_indices[i]:gradient_indices[i + 1]].mean() < cutoff:
                    crossing_indices.append((gradient_indices[i] + gradient_indices[i + 1]) // 2)
                i += 3  # Skip to the next pair of gradient indices
            else:
                i += 1

        return crossing_indices




    def process_sectioned_cutoff_AGL_CLI(self, df, filepath=None, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD, min_time_steps=MIN_TIME_STEPS_PER_FILE, crossing_indices=[]):
        """Process the AltitudeAGL column and split the flight based on altitude and time thresholds."""
        # Ensure AltitudeAGL is numeric and remove rows with NaN values
        df['AltitudeAGL'] = pd.to_numeric(df['AltitudeAGL'], errors='coerce')
        df = df.dropna(subset=['AltitudeAGL'])

        original_file_name = os.path.splitext(os.path.basename(filepath))[0]

        df_list = []
        # Store the indices where the AltitudeAGL crosses the cutoff

        if not crossing_indices:
            crossing_indices = self.get_crossing_indices(df, cutoff=cutoff, time_threshold=time_threshold)



        # Ensure we sort the crossing indices in ascending order
        crossing_indices = sorted(crossing_indices)

        # Include the last index of the df in the crossing indices


        # Check if the crossing indices are correct by removing any indices where the AltitudeAGL is more than 80 ft from the cutoff
        # crossing_indices = [index for index in crossing_indices if abs(df['AltitudeAGL'].iloc[index] - cutoff) < 80]

        # Save the sections to a new csv file with _section_{n} appended to the original file name
        section_counter = 1
        print("Length of crossing indices:", len(crossing_indices))
        for i in range(len(crossing_indices) - 1):
            start_index = int(crossing_indices[i])
            end_index = int(crossing_indices[i + 1])
            print(f"Processing section from index {start_index} to {end_index}...")
            section_df = df.iloc[start_index:end_index]
            # section_df = section_df[section_df['AltitudeAGL'] > cutoff]  # Ensure we keep only values above the cutoff
            
            if len(section_df) > min_time_steps:
                df_list.append((section_df, start_index, end_index))
            else:
                print(f"Skipping section from index {start_index} to {end_index} due to insufficient time steps...")
                continue

        print(f"Found {len(df_list)} sections in {original_file_name}...")
        self.plot_joint_altitude_profile_CLI(df, df_list, original_file_name, cutoff, crossing_indices)

        save = input("Save the sections based on the cutoff? (y/n): ")
        if save.lower() == 'y':
            for section_df, _, _ in df_list:
                section_file_path = f"{self.save_folder}/{original_file_name}_section_{section_counter}.csv"
                section_df.to_csv(section_file_path, index=False)
                print(f"Saved section {section_counter} to {section_file_path}...")
                section_counter += 1

        else:
            rerun_with_new_cutoff = input("Rerun the function with a new cutoff and time threshold? (y/n): ")
            if rerun_with_new_cutoff.lower() == 'y':
                inject_index = input("Manually inject an index in the crossing indices? (y/n): ")
                if inject_index.lower() == 'y':
                    new_index = int(input("Enter the index to inject: "))
                    crossing_indices.append(new_index)

                    self.process_sectioned_cutoff_AGL_CLI(df, filepath, cutoff=cutoff, time_threshold=time_threshold, crossing_indices=crossing_indices)

                user_input = input("Enter the new cutoff value, time threshold value, and minimum time steps: ").strip()
                if user_input:
                    new_cutoff, new_time_threshold, new_min_time_steps = map(int, user_input.split())
                else:
                    new_cutoff, new_time_threshold, new_min_time_steps = cutoff, time_threshold, min_time_steps
                
                if new_cutoff is None:
                    new_cutoff = cutoff

                if new_time_threshold is None:
                    new_time_threshold = time_threshold

                if new_min_time_steps is None:
                    new_min_time_steps = min_time_steps

                self.process_sectioned_cutoff_AGL_CLI(df, filepath, cutoff=new_cutoff, time_threshold=new_time_threshold, min_time_steps=new_min_time_steps)

            return df_list

    


    def process_sectioned_cutoff_AGL_GUI(self, df, filepath=None, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD, min_time_steps=MIN_TIME_STEPS_PER_FILE, crossing_indices=[]):
        """Process the AltitudeAGL column and split the flight based on altitude and time thresholds."""
        # Ensure AltitudeAGL is numeric and remove rows with NaN values
        df['AltitudeAGL'] = pd.to_numeric(df['AltitudeAGL'], errors='coerce')
        df = df.dropna(subset=['AltitudeAGL'])

        original_file_name = os.path.splitext(os.path.basename(filepath))[0]

        df_list = []
        # Store the indices where the AltitudeAGL crosses the cutoff

        if not crossing_indices:
            crossing_indices = self.get_crossing_indices(df, cutoff=cutoff, time_threshold=time_threshold)


        # Ensure we sort the crossing indices in ascending order
        crossing_indices = sorted(crossing_indices)
        logging.info(f'Using crossing indices: {crossing_indices}')

        # Check if the crossing indices are correct by removing any indices where the AltitudeAGL is more than 80 ft from the cutoff
        # crossing_indices = [index for index in crossing_indices if abs(df['AltitudeAGL'].iloc[index] - cutoff) < 80]

        # Save the sections to a new csv file with _section_{n} appended to the original file name
        logging.info(f"Length of crossing indices: {len(crossing_indices)}")
        for i in range(len(crossing_indices) - 1):
            start_index = int(crossing_indices[i])
            end_index = int(crossing_indices[i + 1])
            logging.info(f"Processing section from index {start_index} to {end_index}...")
            section_df = df.iloc[start_index:end_index]
            
            if len(section_df) > min_time_steps:
                df_list.append((section_df, start_index, end_index))
            else:
                logging.info(f"Skipping section from index {start_index} to {end_index} due to insufficient time steps...")
                continue

        logging.info(f"Found {len(df_list)} sections in {original_file_name}...")
        self.plot_joint_altitude_profile_GUI(df, df_list, original_file_name, cutoff, crossing_indices)

        return self.num_sections_saved # Return the number of sections processed
    

    def plot_joint_altitude_profile_GUI(self, df, df_list, original_file_name, cutoff, crossing_indices):
        """Plot the joint altitude profile for the entire flight and the sections."""
        fig, ax = plt.subplots(figsize=(18, 5))

        # Plot the original AltitudeAGL
        ax.plot(df.index, df['AltitudeAGL'], label='Altitude AGL original', color='blue')
        ax.axhline(y=cutoff, color='orange', linestyle='--', label=f'Cutoff = {cutoff} ft')
        ax.set_title(f"Altitude AGL for {original_file_name}")
        ax.set_xlabel('Index')
        ax.set_ylabel('Altitude AGL (ft)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        

        # Plot the sections and vertical lines at crossing indices
        for section_df, start_index, end_index in df_list:
            ax.plot(section_df.index, section_df['AltitudeAGL'], color='red', label='Altitude AGL considered')
            ax.axvline(x=end_index, color='red', linestyle='--')

        # Add original indices from df_list
        for section_df, start_index, end_index in df_list:
            ax.text(start_index, cutoff, f'{start_index}', color='purple', fontsize=10, verticalalignment='bottom', horizontalalignment='right')


        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # plt.show()

                # Create sliders for the thresholds
        axcolor = 'lightgoldenrodyellow'
        ax_min_time_steps = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
        ax_cutoff = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        # ax_time_threshold = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
        ax_save_button = plt.axes([0.8, 0.9, 0.1, 0.05])

        slider_min_time_steps = Slider(ax_min_time_steps, 'Min Time Steps', 1, 1000, valinit=MIN_TIME_STEPS_PER_FILE, valstep=10)
        slider_cutoff = Slider(ax_cutoff, 'Altitude Cutoff', 0, 5000, valinit=CUTOFF, valstep=1)
  #      slider_time_threshold = Slider(ax_time_threshold, 'Time Threshold', 0, 2000, valinit=TIME_THRESHOLD, valstep=10)
        save_button = Button(ax_save_button, 'Save Sections')
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust space equally on all sides

        def update(val, save_frame=False):
            nonlocal df_list
            if save_frame:
                global frame_count

            min_time_steps = slider_min_time_steps.val
            cutoff = slider_cutoff.val
#            time_threshold = slider_time_threshold.val

            # Clear the current plot
            ax.clear()

            # Recalculate the crossing indices and sections
            crossing_indices = self.get_crossing_indices(df, cutoff=cutoff) # can include time threshold here

            crossing_indices = sorted(crossing_indices)


            df_list = []
            for i in range(len(crossing_indices) - 1):
                start_index = int(crossing_indices[i])
                end_index = int(crossing_indices[i + 1])
                section_df = df.iloc[start_index:end_index]
                if len(section_df) > min_time_steps:
                    df_list.append((section_df, start_index, end_index))

            # Plot the updated data
            ax.plot(df.index, df['AltitudeAGL'], label='Altitude AGL original', color='blue')
            ax.axhline(y=cutoff, color='orange', linestyle='--', label=f'Cutoff = {cutoff} ft')
            ax.set_title(f"Altitude AGL for {original_file_name}")

            ax.set_ylabel('Altitude AGL (ft)')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

            for section_df, start_index, end_index in df_list:
                ax.plot(section_df.index, section_df['AltitudeAGL'], color='red', label='Altitude AGL considered')
                ax.axvline(x=end_index, color='red', linestyle='--')

            for crossing_index in crossing_indices:
                ax.axvline(x=crossing_index, color='green', linestyle='--', label='Crossing Index = (Start + End) / 2')

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())


            if save_frame:
                # Save frame
                frame_path = f"{frames_dir}/frame_{frame_count:03d}.png"
                plt.savefig(frame_path)
                frame_count += 1

            fig.canvas.draw_idle()

        def save_sections(event):
            nonlocal df_list

            section_counter = 0


            for section_df, _, _ in df_list:
                
                match_1 = re.search(r'_before_(\d+)', original_file_name)
                match_2 = re.search(r'_day_of_(\d+)', original_file_name)
                match_3 = re.search(r'_after_(\d+)', original_file_name)

                if match_1:
                    before_number = int(match_1.group(1))
                    file_name = f"event_{self.EVENT_STATE['event_id']}_before_{before_number}_section_{section_counter}"
                    section_file_path = f"{self.save_folder}/{file_name}.csv"
                    section_df.to_csv(section_file_path, index=False)

                    # print("Before...")

                elif match_2:
                   
                    # print("Day of.....")
                    day_of_number = int(match_2.group(1))
                    file_name = f"event_{self.EVENT_STATE['event_id']}_day_of_{day_of_number}_section_{section_counter}"
                    section_file_path = f"{self.save_folder}/{file_name}.csv"
                    section_df.to_csv(section_file_path, index=False)

                elif match_3:
                    
                    # print("After....")
                    after_number = int(match_3.group(1))
                    file_name = f"event_{self.EVENT_STATE['event_id']}_after_{after_number}_section_{section_counter}"
                    section_file_path = f"{self.save_folder}/{file_name}.csv"
                    section_df.to_csv(section_file_path, index=False)
                    

                else:
                    print("No match found...")
                    section_file_path = None

                if section_file_path:
                    logging.info(f"\033[95mSaved section {section_counter} to {section_file_path}...\033[0m")




                original_file_path = os.path.join(self.folder_path, original_file_name) 

                save_last_processed_file(original_file_path, LAST_PROCESSED_FILE_PATH) #save the original file path to be able to sort the files later
                logging.info(f"\033[94mUpdated last processed file to {original_file_path}...\033[0m")
                section_counter += 1

            #save the parameters (cutoff, time_threshold, min_time_steps) to a file
#            params = {'Cutoff': slider_cutoff.val, 'Time Threshold': slider_time_threshold.val, 'Min Time Steps': slider_min_time_steps.val}
#            save_parameters_to_file(params, PARAMS_SAVE_FILE_PATH)
            logging.info(f"\033[94mSaved parameters to {slider_cutoff.val}, {slider_min_time_steps.val} to {PARAMS_SAVE_FILE_PATH}...\033[0m")
            plt.close(fig)
            return section_counter # when the sections are saved, return the number of sections saved

        slider_min_time_steps.on_changed(update)
        slider_cutoff.on_changed(update)
#        slider_time_threshold.on_changed(update)
        save_button.on_clicked(lambda event: setattr(self, 'num_sections_saved', save_sections(event))) # Save the number of sections saved

        plt.show()

    def plot_joint_altitude_profile_CLI(self, df, df_list, original_file_name, cutoff, crossing_indices):
        """Plot the joint altitude profile for the entire flight and the sections."""
        fig, ax = plt.subplots(figsize=(18, 5))

        # Plot the original AltitudeAGL
        ax.plot(df.index, df['AltitudeAGL'], label='Altitude AGL original', color='blue')
        ax.axhline(y=cutoff, color='gray', linestyle='--', label=f'Cutoff = {cutoff} ft')
        ax.set_title(f"Altitude AGL for {original_file_name}")
        ax.set_xlabel('Index')
        ax.set_ylabel('Altitude AGL (ft)')
        
        ax.text(0.95, 0.95, f'Time Threshold = {TIME_THRESHOLD} s', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Plot the sections and vertical lines at crossing indices
        for section_df, start_index, end_index in df_list:
            ax.plot(section_df.index, section_df['AltitudeAGL'], color='red', label='Altitude AGL considered')
            ax.axvline(x=start_index, color='red', linestyle='--', label='Estimated flight sections')
            ax.axvline(x=end_index, color='red', linestyle='--')

        # Add original indices from df_list
        for section_df, start_index, end_index in df_list:
            ax.axvline(x=start_index, color='purple', linestyle='--', label='Index')
            ax.axvline(x=end_index, color='orange', linestyle='--', label='Index')
            ax.text(start_index, cutoff, f'{start_index}', color='purple', fontsize=10, verticalalignment='bottom', horizontalalignment='right')
            ax.text(end_index, cutoff, f'{end_index}', color='orange', fontsize=10, verticalalignment='bottom', horizontalalignment='right')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.show()



            
                    
    def set_file_paths(self, file_paths):
        self.file_paths = file_paths


    

def main():

#### ----------------- execution code for GUI separator ----------------- ####




    # load_folder_path = "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/c37_cleaned_all_agl" # path where the csv files are to be loaded from
    # save_folder_path = f"/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/c37_cleaned_all_agl_separated" # path where the processed data is to be saved
    # preprocessor = NGAFIDEventsPreprocessor(load_folder_path, save_folder_path)
    


    # # # Get all CSV files in the folder
    all_files = glob.glob(os.path.join(LOAD_FOLDER_PATH, '*'))
    sorted_files = sorted(all_files, key=extract_keys) # Sort the files based on the date, before, day_of, after, number in that order
    for i in sorted_files[:20]:
        print(i)


    # processed_files = glob.glob(os.path.join(save_folder_path, '*'))
    # last_processed_file = get_last_processed_file()
    # start_index = 0
    # if last_processed_file:
    #     if last_processed_file in sorted_files:
    #         start_index = sorted_files.index(last_processed_file) + 1
    #         logging.info(f"\033[92mLast processed file: {last_processed_file} at index {start_index}...\033[0m")
    #     else:
    #         logging.error(f"\033[91mLast processed file: {last_processed_file} not found in all_files.\033[0m")
    #         #print a few files in all_files
    #         start_index = 0

    
    # print(f"Starting processing from index {start_index}...")
    # for file_path in all_files[start_index:]:
    #     print("---------------------------------")
    #     #if the file is a txt file, copy it to the save folder
    #     if not file_path.endswith('.csv'):
    #         if  file_path.endswith('.py'):
    #             save_path = os.path.join(save_folder_path, os.path.basename(file_path)) # save the file in the save folder with the same name
    #             print(f"Copying {os.path.basename(file_path)} to {save_path}...")
    #             with open(file_path, 'r') as f_src:
    #                 with open(save_path, 'w') as f_dst:
    #                     f_dst.write(f_src.read())

    #         if file_path.endswith('.txt'):
    #             save_path = os.path.join(save_folder_path, os.path.basename(file_path))
    #     else:
    #         print(f"Processing {os.path.basename(file_path)}...")
    #         df = preprocessor.load_data(file_path, skiprows=0, columns_to_drop=COLUMNS_TO_DROP)
    #         # preprocessor.process_cutoff_AGL(df, file_path)
    #         preprocessor.process_sectioned_cutoff_AGL_GUI(df, file_path, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD)

    # # Save the last processed file
    # # save_last_processed_file(all_files[-1])

    # if SAVE_FRAMES_AS_GIF:
    #     frames = sorted([f"{frames_dir}/{file}" for file in os.listdir(frames_dir) if file.endswith(".png")])
    #     images = [imageio.imread(frame) for frame in frames]
    #     imageio.mimsave("demo.gif", images, duration=0.3)  # Adjust duration for speed
    #     print("GIF saved as demo.gif")




    # #####----- testing with GUI ----####

    # load_fake_file_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/fake_files'
    # store_fake_file_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/fake_files_processed'
    # SAVE_FRAMES_AS_GIF = False

    # # load_folder_path = "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/c37_cleaned_all_agl" # path where the csv files are to be loaded from
    # # save_folder_path = f"/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/c37_cleaned_all_agl_separated_{CUTOFF}" # path where the processed data is to be saved
    # preprocessor = NGAFIDEventsPreprocessor(load_folder_path= load_fake_file_path, save_folder_path=store_fake_file_path)
    


    # # # # Get all CSV files in the folder
    # all_files = glob.glob(os.path.join(load_fake_file_path, '*'))
    # processed_files = glob.glob(os.path.join(store_fake_file_path, '*'))
    # sorted_files = sorted(all_files, key=extract_keys) # Sort the files based on the date, before, day_of, after, number in that order
    # # for i in sorted_files[:20]:
    # #     print(i)

    # last_processed_file = get_last_processed_file(LAST_PROCESSED_FILE_PATH)
    # start_index = 0
    # if last_processed_file:
    #     if last_processed_file in processed_files:
    #         start_index = processed_files.index(last_processed_file) + 1
    #         logging.info(f"\033[92mLast processed file: {last_processed_file} at index {start_index}...\033[0m")
    #     else:
    #         logging.error(f"\033[91mLast processed file: {last_processed_file} not found in all_files.\033[0m")
    #         #print a few files in all_files
    #         start_index = 0

    
    # print(f"Starting processing from index {start_index}...")
    # for file_path in all_files[start_index:]:
    #     print("---------------------------------")
    #     #if the file is a txt file, copy it to the save folder
    #     if file_path.endswith('.txt') or file_path.endswith('.py'):
    #         save_path = os.path.join(store_fake_file_path, os.path.basename(file_path)) # save the file in the save folder with the same name
    #         print(f"Copying {os.path.basename(file_path)} to {save_path}...")
    #         with open(file_path, 'r') as f_src:
    #             with open(save_path, 'w') as f_dst:
    #                 f_dst.write(f_src.read())
    #     else:
    #         print(f"Processing {os.path.basename(file_path)}...")
    #         df = preprocessor.load_data(file_path, skiprows=0, columns_to_drop=COLUMNS_TO_DROP)
    #         # preprocessor.process_cutoff_AGL(df, file_path)
    #         preprocessor.process_sectioned_cutoff_AGL_GUI(df, file_path, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD)

    # # Save the last processed file
    # # save_last_processed_file(all_files[-1])

    # if SAVE_FRAMES_AS_GIF:
    #     frames = sorted([f"{frames_dir}/{file}" for file in os.listdir(frames_dir) if file.endswith(".png")])
    #     images = [imageio.imread(frame) for frame in frames]
    #     imageio.mimsave("demo.gif", images, duration=0.3)  # Adjust duration for speed
    #     print("GIF saved as demo.gif")










    ############--- testing the process_cutoff_AGL function on a fake file ---############
    # Run the process_cutoff_AGL function on the fake_file.csv
#     load_fake_file_path = '/Users/manasdubey2022/Desktop/fake_files'
#     store_fake_file_path = '/Users/manasdubey2022/Desktop/fake_files_processed'

#     csv_files = glob.glob(os.path.join(load_fake_file_path, '*'))
#     plot_altitude_profiles(csv_files, n_files=1)
    
# #    if store folder exists, delete it
#     if os.path.exists(store_fake_file_path):
#         print(f"Deleting {store_fake_file_path}...")
#         os.system(f"rm -r {store_fake_file_path}") #  !!!!!! Use with caution
#     preprocessor = DayOfEventsPreprocessor(load_folder_path= load_fake_file_path, save_folder_path=store_fake_file_path)

#     for file_path in csv_files:
#         print("---------------------------------")
#         print(f"Processing {os.path.basename(file_path)}...")
#         df = preprocessor.load_data(file_path, skiprows=0)
#         preprocessor.process_sectioned_cutoff_AGL(df, file_path)
    


    

#     fake_csv_files = glob.glob(os.path.join(store_fake_file_path, '*'))
#     plot_altitude_profiles(fake_csv_files, n_files=10)


if __name__ == '__main__':
    main()  












    


    





