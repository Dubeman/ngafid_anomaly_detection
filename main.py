import os
import glob
import logging
import imageio
from NGAFID_events_processor import NGAFIDEventsPreprocessor
import json
from utils import extract_keys, get_last_processed_file, create_work_order_event, load_config, update_work_order_event
import re
import uuid
config = load_config('config.json')


logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COLUMNS_TO_DROP = config['COLUMNS_TO_DROP']
MIN_TIME_STEPS_PER_FILE = config['MIN_TIME_STEPS_PER_FILE']
CUTOFF = config['CUTOFF']
TIME_THRESHOLD = config['TIME_THRESHOLD']
SAVE_FRAMES_AS_GIF = config['SAVE_FRAMES_AS_GIF']
LAST_PROCESSED_FILE_PATH = config['LAST_PROCESSED_FILE_PATH']
PARAMS_SAVE_FILE_PATH = config['PARAMS_SAVE_FILE_PATH']
FRAMES_DIR = config['FRAMES_DIR']
EVENT_DEFINITION = config['EVENT_DEFINITION']
LOAD_FOLDER_PATH = config['LOAD_FOLDER_PATH']  # path where the csv files are to be loaded from
SAVE_FRAMES_AS_GIF = config['SAVE_FRAMES_AS_GIF'] # whether to save the frames as a gif (TRUE/FALSE)
SAVE_FOLDER_PATH = config['SAVE_FOLDER_PATH'] # path where the processed data is to be saved
GIF_SAVE_FOLDER_PATH = config['GIF_SAVE_FOLDER_PATH'] # path where the gif is to be saved
CLUSTER_EVENTS_FILE_PATH = config['CLUSTER_EVENTS_FILE_PATH'] # path to the file containing the cluster events

EVENT_STATE = None
DF_SECTIONS_DICT = {
    'before_flights': [],
    'day_of_flights': [],
    'after_flights': []
}



def update_df_sections_dict(file_path, num_sections, df_sections_dict, EVENT_STATE):
    '''
    Update the df_sections_dict with the flight names for the given file.
    '''

    match_1 = re.search(r'_before_(\d+)', file_path)
    match_2 = re.search(r'_day_of_(\d+)', file_path)
    match_3 = re.search(r'_after_(\d+)', file_path)

    for i in range(num_sections):


        if match_1:
            before_number = int(match_1.group(1))
            filename = f"event_{EVENT_STATE['event_id']}_before_{before_number}_section_{i}.csv"
            df_sections_dict['before_flights'].append(filename)

        elif match_2:
            day_of_number = str(match_2.group(1))[-2:]
            filename = f"event_{EVENT_STATE['event_id']}_day_of_{day_of_number}_section_{i}.csv"
            df_sections_dict['day_of_flights'].append(filename)

        elif match_3:
            after_number = int(match_3.group(1))
            filename = f"event_{EVENT_STATE['event_id']}_after_{after_number}_section_{i}.csv"
            df_sections_dict['after_flights'].append(filename)

        else:
            
            print(f"File {file_path} does not contain any of the keywords: before, day_of, after.")
        
    
    




def launch_gui():
    #### ----------------- execution code for GUI separator ----------------- ####

    global EVENT_STATE, DF_SECTIONS_DICT

    preprocessor = NGAFIDEventsPreprocessor(LOAD_FOLDER_PATH, SAVE_FOLDER_PATH)
    


    # # # Get all CSV files in the folder
    all_files = glob.glob(os.path.join(LOAD_FOLDER_PATH, '*'))
    processed_files = glob.glob(os.path.join(SAVE_FOLDER_PATH, '*'))
    sorted_files = sorted(all_files, key=extract_keys) # Sort the files based on the date, before, day_of, after, number in that order
    

    # for i in range(len(all_files)):
    #     print(f"{i}: {all_files[i]}")

    last_processed_file = get_last_processed_file(LAST_PROCESSED_FILE_PATH)
    start_index = 0
    if last_processed_file:
        if last_processed_file in sorted_files:
            start_index = sorted_files.index(last_processed_file) + 1
            logging.info(f"\033[92mLast processed file: {last_processed_file} at index {start_index}...\033[0m")
        else:
            logging.error(f"\033[91mLast processed file: {last_processed_file} not found in all_files.\033[0m")
            #print a few files in all_files
            start_index = 0


    print(f"Starting processing from index {start_index}...")
    ### ALWAYS SAVE TILL THE NEXT .TXT FILE IS ENCOUNTERED OTHERWISE THE EVENT STATE WILL BE LOST###
    ### ALWAYS SAVE TILL THE NEXT .TXT FILE IS ENCOUNTERED OTHERWISE THE EVENT STATE WILL BE LOST###
    ### ALWAYS SAVE TILL THE NEXT .TXT FILE IS ENCOUNTERED OTHERWISE THE EVENT STATE WILL BE LOST###
    

    for file_path in sorted_files[start_index:]:
        
        print("\033[91m\033[1m### ALWAYS SAVE TILL THE NEXT .TXT FILE IS ENCOUNTERED OTHERWISE THE EVENT STATE WILL BE LOST###\033[0m")
        print("---------------------------------")

        if not file_path.endswith('.csv'):


            if file_path.endswith('.txt'):


                if EVENT_STATE: # encountered a txt file i.e. a new work order event, df_sections_dict must be filled up
                    update_work_order_event(EVENT_STATE, DF_SECTIONS_DICT, CLUSTER_EVENTS_FILE_PATH, file_path)
                    logging.info(f"\033[93mClosed work order event for {os.path.basename(file_path)}\033[0m")
                    EVENT_STATE = None
                    for key in DF_SECTIONS_DICT: # clear the df_sections_dict
                        DF_SECTIONS_DICT[key].clear()
                    preprocessor.num_sections_saved = 0


            else:
                print(f"Skipping {os.path.basename(file_path)}...")
        else:
            if not EVENT_STATE:
                EVENT_STATE = create_work_order_event(EVENT_DEFINITION, file_path)
                logging.info(f"\033[92mCreated work order event for {os.path.basename(file_path)}\033[0m")
                preprocessor.EVENT_STATE = EVENT_STATE

            df = preprocessor.load_data(file_path, skiprows=0, columns_to_drop=COLUMNS_TO_DROP)
            num_sections = preprocessor.process_sectioned_cutoff_AGL_GUI(df, file_path, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD)
            update_df_sections_dict(file_path, num_sections, DF_SECTIONS_DICT , EVENT_STATE)
            logging.info(f"\033[92mProcessed {os.path.basename(file_path)}\033[0m")







    if SAVE_FRAMES_AS_GIF:
        frames = sorted([f"{FRAMES_DIR}/{file}" for file in os.listdir(FRAMES_DIR) if file.endswith(".png")])
        images = [imageio.imread(frame) for frame in frames]
        gif_save_path = os.path.join(GIF_SAVE_FOLDER_PATH, 'output.gif')
        imageio.mimsave(gif_save_path, images, duration=0.3)  # Adjust duration for speed
        print("GIF saved as demo.gif")

def process_auto():
    '''
    Processes files in the specified load folder, applies preprocessing, and saves the results in the save folder.
    This function performs the following steps:
    1. Initializes the NGAFIDEventsPreprocessor with the load and save folder paths.
    2. Retrieves all files from the load folder and sorts them.
    3. Determines the starting index based on the last processed file.
    4. Iterates over the sorted files starting from the determined index.
    5. For each file:
        - If the file is a .txt file, updates and closes the work order event.
        - If the file is a .csv file, creates a new work order event and processes the data.
    6. Optionally saves the processed frames as a GIF if the SAVE_FRAMES_AS_GIF flag is set.
    Global Variables:
    - EVENT_STATE: The current state of the event being processed.
    - DF_SECTIONS_DICT: A dictionary to store sections of data frames.
    Raises:
    - json.decoder.JSONDecodeError: If there is an error decoding a JSON file.
    Note:
    - The function uses various global constants and paths that need to be defined elsewhere in the code.
    - The function updates the EVENT_STATE and DF_SECTIONS_DICT global variables as it processes the files.
    - The function prints log messages to the console to indicate the progress and status of the processing.
    '''
    global EVENT_STATE, DF_SECTIONS_DICT

    preprocessor = NGAFIDEventsPreprocessor(LOAD_FOLDER_PATH, SAVE_FOLDER_PATH)

    all_files = glob.glob(os.path.join(LOAD_FOLDER_PATH, '*'))
    processed_files = glob.glob(os.path.join(SAVE_FOLDER_PATH, '*'))
    sorted_files = sorted(all_files, key=extract_keys)

    last_processed_file = get_last_processed_file(LAST_PROCESSED_FILE_PATH)
    start_index = 0
    if last_processed_file:

        if last_processed_file in sorted_files:
            start_index = sorted_files.index(last_processed_file) + 1
            logging.info(f"\033[92mLast processed file: {last_processed_file} at index {start_index}...\033[0m")
        else:
            logging.error(f"\033[91mLast processed file: {last_processed_file} not found in all_files.\033[0m")
            start_index = 0

    print(f"Starting processing from index {start_index}...")
    i = 0
    for file_path in sorted_files[start_index:]:

        print("---------------------------------")
        if not file_path.endswith('.csv'):

            try:
                if file_path.endswith('.txt'):
                    if EVENT_STATE:
                        update_work_order_event(EVENT_STATE, DF_SECTIONS_DICT, CLUSTER_EVENTS_FILE_PATH, file_path)
                        logging.info(f"\033[93mClosed work order event for {os.path.basename(file_path)}\033[0m")
                        EVENT_STATE = None
                        for key in DF_SECTIONS_DICT:
                            DF_SECTIONS_DICT[key].clear()
                        preprocessor.num_sections_saved = 0
            
                else:
                    print(f"Skipping {os.path.basename(file_path)}...")

            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError in file {file_path}: {e}")
                raise e
        else:
            if not EVENT_STATE:
                EVENT_STATE = create_work_order_event(EVENT_DEFINITION, file_path)
                logging.info(f"\033[92mCreated work order event for {os.path.basename(file_path)}\033[0m")
                preprocessor.EVENT_STATE = EVENT_STATE

            df = preprocessor.load_data(file_path, skiprows=0, columns_to_drop=COLUMNS_TO_DROP)
            num_sections = preprocessor.process_cutoff_AGL(df, file_path, cutoff=CUTOFF, time_threshold=TIME_THRESHOLD)
            update_df_sections_dict(file_path, num_sections, DF_SECTIONS_DICT, EVENT_STATE)
            logging.info(f"\033[92mProcessed {os.path.basename(file_path)}\033[0m")
        i+=1

    if SAVE_FRAMES_AS_GIF:
        frames = sorted([f"{FRAMES_DIR}/{file}" for file in os.listdir(FRAMES_DIR) if file.endswith(".png")])
        images = [imageio.imread(frame) for frame in frames]
        imageio.mimsave(GIF_SAVE_FOLDER_PATH, images, duration=0.3)
        print("GIF saved as demo.gif")


def main():
    launch_gui()
    # process_auto()

if __name__ == "__main__":
    main()