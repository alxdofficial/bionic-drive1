import json
import os
import re

def extract_time_from_filename(filename):
    # Extract the timestamp from the filename
    parts = filename.split('__')[0]  # Gets "n015-2018-07-18-11-50-34+0800"
    time_part = parts.split('-')[1:]  # Extracts the date and time part
    return '-'.join(time_part)  # Return the time as a sortable string

def extract_number_from_filename(filename):
    # Extract the numerical part from the filename (e.g., "1531886015447423" from "n015-2018-07-18-11-50-34+0800__CAM_BACK__1531886015447423.jpg")
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

def sort_json_by_time_and_number(json_file, output_file):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Sort the data by the timestamp and the numerical part of the CAM_FRONT image filename
    data.sort(key=lambda x: (
        extract_time_from_filename(x[1]["CAM_FRONT"]),
        extract_number_from_filename(x[1]["CAM_FRONT"])
    ))

    # Save the sorted data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Sorted JSON file saved to {output_file}")

if __name__ == "__main__":
    # Define your input and output files
    input_json_file = 'data/inference/sorted_multi_frame_test.json'  # Replace with your input file path
    output_json_file = 'data/inference/sorted_multi_frame_test.json'  # Replace with your desired output file path

    # Sort the JSON file by time and image number
    sort_json_by_time_and_number(input_json_file, output_json_file)
