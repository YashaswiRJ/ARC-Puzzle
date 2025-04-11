import json

# Read the raw JSON file
with open("ques3.json", "r") as file:
    data = json.load(file)  # Load JSON into a Python dictionary

# Convert "\n" strings to actual newlines
formatted_json = json.dumps(data, indent=4)
formatted_json = formatted_json.replace("\\n", "\n")  # Fix newlines

# Print formatted JSON
print(formatted_json)
