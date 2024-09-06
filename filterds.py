import json
import re

def extract_coordinates(text):
    # Regex to extract coordinates in the format <c1,CAM_FRONT_RIGHT,215.8,664.2>
    return re.findall(r'<c\d+,\w+,\d+\.\d+,\d+\.\d+>', text)

def filter_dataset(input_file, output_file):
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_data = []

    for entry in data:
        question = entry[0]['Q']
        answer = entry[0]['A']

        # Extract coordinates from question and answer
        question_coords = set(extract_coordinates(question))
        answer_coords = set(extract_coordinates(answer))

        # Check if all coordinates in the answer are also in the question
        # or if there are no coordinates in the answer
        if answer_coords.issubset(question_coords) or not answer_coords:
            filtered_data.append(entry)

    # Save the filtered dataset
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/inference/sorted_multi_frame_val.json'  # Replace with your input file path
    output_file = 'data/inference/filtered_sorted_multi_frame_val.json'  # Replace with your desired output file path

    filter_dataset(input_file, output_file)
