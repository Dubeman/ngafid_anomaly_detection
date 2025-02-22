import json
import os

def calculate_total_files(json_file_path, directory_files):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    total_files = 0
    
    for event in data:
        event['before_flights'] = [f for f in event.get('before_flights', []) if f in directory_files]
        event['day_of_flights'] = [f for f in event.get('day_of_flights', []) if f in directory_files]
        event['after_flights'] = [f for f in event.get('after_flights', []) if f in directory_files]
        
        total_files += len(event['before_flights'])
        total_files += len(event['day_of_flights'])
        total_files += len(event['after_flights'])
    json_file_path_altered = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/metadata/c37_cluster_events_altered.json'
    with open(json_file_path_altered, 'w') as file:
        json.dump(data, file, indent=4)
    
    return total_files

def count_files_in_directory(directory_path):
    return [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))]

if __name__ == "__main__":
    json_file_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/metadata/c37_cluster_events.json'
    directory_path = '/Users/manasdubey2022/Desktop/NGAFID_Data_Processor/data/c37_cleaned'
    
    directory_files = count_files_in_directory(directory_path)
    total_files = calculate_total_files(json_file_path, directory_files)
    directory_file_count = len(directory_files)
    
    print(f"Total number of files: {total_files}")
    print(f"Number of files in directory: {directory_file_count}")
    
    if total_files == directory_file_count:
        print("The sum of the total files is equal to the number of files in the directory.")
    else:
        print("The sum of the total files is not equal to the number of files in the directory.")