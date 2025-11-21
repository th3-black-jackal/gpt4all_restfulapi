import json
import sys
def read_json_to_dict(file_path):
    """
    Reads a JSON file and returns its contents as a Python dictionary.
    Includes error handling for common issues.

    :param file_path: Path to the JSON file.
    :return: Dictionary representation of the JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except PermissionError:
        print(f"Error: Permission denied when accessing '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None