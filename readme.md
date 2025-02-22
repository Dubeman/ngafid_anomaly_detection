# NGAFID Data Processor  

## Overview  
The NGAFID Data Processor is a Python-based repo designed to process and analyze flight data from the National General Aviation Flight Information Database (NGAFID). It enables users to parse, clean, and visualize flight data to effectively identify patterns and anomalies.  

## Features  

The repository includes `NGAFID_events_processor.py`, a script for preprocessing and analyzing flight data. The key functionalities include:  

- **Parsing flight data** from CSV files.  
- **Data cleaning & preprocessing**, including type conversion, NaN removal, and validity filtering.  
- **Visualization tools** to plot altitude profiles for entire flights and specific segments.  
- **Flight segmentation**, identifying valid segments based on altitude and time thresholds.  

## Configuration  

The **`config.json`** file allows customization of data processing parameters to tailor the analysis.  

### Threshold Definitions  

- **CUTOFF** – The minimum altitude required for a flight segment to be considered valid.  
- **TIME_THRESHOLD** – The minimum duration for a valid flight segment. (Similar to `MIN_TIME_STEPS_PER_FILE`, originally used in a different algorithm.)  
- **Other Parameters** – Additional settings like speed thresholds and event markers.  

## Installation  

Ensure you have Python installed, then install the required dependencies using:  

```bash
pip install -r requirements.txt
```  

## Usage

The main script to run the application is `main.py`. It includes methods to launch the GUI and process data automatically:

- **launch_gui**: This method starts the graphical user interface for interactive data processing.
- **process_auto**: This method processes the data automatically based on predefined configurations.

To run the application, use the following command:

```bash
python main.py
```

## File Checks
The repository includes `file_checks.py` for testing flight data files:

- **File Integrity Tests**: Checks for corruption and validates columns and data types.
- **Data Consistency Checks**: Ensures data adheres to rules.
- **Quick Visualization**: Visualizes time splits to identify irregularities.

Run the checks with:

```bash
python file_checks.py
```

