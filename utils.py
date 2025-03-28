import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import uuid
from IPython.display import display






def plot_altitude_profiles(csv_files, n_files=10):
    # Calculate the number of rows and columns for the subplot grid
    nrows = (n_files + 1) // 2
    ncols = 2

    # Create a tile-based plot for all files in csv_files
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

    # If there's only one row, axs will be 1-dimensional
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, file in enumerate(csv_files[:n_files]):
        df = pd.read_csv(file, skiprows=0)
        # Strip the leading and trailing whitespaces from the column names
        df.columns = df.columns.str.strip()
        # Convert 'AltitudeAGL' to numeric, setting errors='coerce' to handle non-numeric values
        df['AltitudeAGL'] = pd.to_numeric(df['AltitudeAGL'], errors='coerce')
        
        # Remove NaN values from 'AltitudeAGL'
        df = df.dropna(subset=['AltitudeAGL'])
        
        row = i // ncols
        col = i % ncols
        axs[row, col].plot(df.index, df['AltitudeAGL'])
        axs[row, col].set_title(os.path.basename(file))
        axs[row, col].set_xlabel('Index')
        axs[row, col].set_ylabel('Altitude AGL (ft)')
        axs[row, col].grid(True)
        # Set Y-axis limits and ticks
        if not df['AltitudeAGL'].empty:
            y_min = df['AltitudeAGL'].min()
            y_max = df['AltitudeAGL'].max()
            y_range = y_max - y_min
            axs[row, col].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            axs[row, col].set_yticks(np.linspace(y_min, y_max, num=10))

    plt.tight_layout()
    plt.show()

def save_last_processed_file(file_name, LAST_PROCESSED_FILE_PATH):
    with open(LAST_PROCESSED_FILE_PATH, 'w') as f:
        f.write(f"{file_name}.csv")

def get_last_processed_file(LAST_PROCESSED_FILE_PATH):
    if os.path.exists(LAST_PROCESSED_FILE_PATH):
        with open(LAST_PROCESSED_FILE_PATH, 'r') as f:
            return f.read().strip()
    return None

def save_parameters_to_file(params, PARAMS_SAVE_FILE_PATH):
        with open(PARAMS_SAVE_FILE_PATH, 'a', newline='') as f:
            writer = csv.writer(f)

            if f.tell() == 0:  # if the file is empty, write the column names
                writer.writerow(params.keys())



            writer.writerow(params.values())

def extract_keys(filename):
        '''
        The sorting key function to extract the date and number from the filename. 
        '''
        date_match = re.search(r'_(\d{4}_\d{2}_\d{2})_', filename)
        number_match = re.search(r'_(\d+)\.csv$', filename)
        date = date_match.group(1) if date_match else '0000_00_00'
        number = int(number_match.group(1)) if number_match else float('inf')
        
        if '_before_' in filename:
            file_type = 0
        elif '_day_of_' in filename:
            file_type = 1
        elif '_after_' in filename:
            file_type = 2
        else:
            file_type = 3  # Default to ensure any other types are sorted last
        
        # Return the negative of the number to sort in reverse order
        
        return date, file_type, number

def load_config(CONFIG_FILE_PATH):
    with open(CONFIG_FILE_PATH, 'r') as f:
        return json.load(f)


def create_work_order_event(EVENT_DEFINITION, filepath=None):
    '''
    Create a JSON for the given work order event.
    '''
    # Read the .txt file contents
    # if filepath:
    #     with open(filepath, 'r') as f:
    #         event_data = json.load(f)
    #         event_id = uuid.uuid4().int  # Generate a random event ID
        
    # else:    
    #     event_id = event_data.get("workorderNumber")

    

    # Create the event using the EVENT_DEFINITION template
    event = EVENT_DEFINITION.copy()
    event.update({
        "event_id":  uuid.uuid4().int,
        "event_cause_text": None,
        "event_action_text": None,
        "event_cluster_label": "c37",
        "before_flights": None,
        "day_of_flights": None,
        "after_flights": None
    })

    return event

def update_work_order_event(event_state, df_names_dict,CLUSTER_EVENTS_FILE_PATH, filepath=None):
    '''
    Update the work order event with the flight data.


    '''


    with open(filepath, 'r') as f:
        event_data = json.load(f)
        print(event_data)
            # event_id = uuid.uuid4().int  # Generate a random event ID
    
    #if "event_id" is a string, convert it to an uuid
    if isinstance(event_state.get("event_id"), str):
        event_state["event_id"] = uuid.uuid4().int



    event_state.update({
        "event_cause_text": event_data.get("cleanedProblem"),
        "event_action_text": event_data.get("originalAction"),
        "before_flights": df_names_dict.get('before_flights'),
        "day_of_flights": df_names_dict.get('day_of_flights'),
        "after_flights": df_names_dict.get('after_flights')
    })

    # Read the existing events from the cluster events file
    if os.path.exists(CLUSTER_EVENTS_FILE_PATH):
        with open(CLUSTER_EVENTS_FILE_PATH, 'r') as f:
            try:
                events = json.load(f)
            except json.JSONDecodeError:
                events = []
    else:
        events = []

    # Append the updated event to the list
    events.append(event_state)

    # Write the updated list of events to the cluster events file
    with open(CLUSTER_EVENTS_FILE_PATH, 'w') as f:
        json.dump(events, f, indent=4)

    



if __name__ == '__main__':
    # config = load_config('config.json')
    # # Define the path to the folder containing the CSV files

    # txt_file = "/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/data/c37_cleaned_all/open_2017_04_17_close_2017_04_17_record.txt"
    # add_json_event(txt_file, config.get('EVENT_DEFINITION'), config.get('CLUSTER_EVENTS_FILE_PATH'))

    df = pd.read_csv("/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/NGAFID_C37_split.csv")
    display(df)


