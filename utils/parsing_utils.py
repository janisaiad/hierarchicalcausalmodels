import re


def source(input_string):
    # Define the regex pattern to match the word part
    pattern = r'^(_?[a-zA-Z]+)\d+(_\d+)?$'

    # Search for the pattern in the input string
    match = re.match(pattern, input_string)

    if match:
        # Return the word part
        return match.group(1)
    else:
        # Return None if the pattern does not match
        return None

def source_sample(input_string):
    # Define the regex pattern to match the desired part
    pattern = r'^(_?[a-zA-Z]+\d+)(_?\d+)?$'

    # Search for the pattern in the input string
    match = re.match(pattern, input_string)

    if match:
        # Return the matched part
        return match.group(1) + (match.group(2) if match.group(2) else '')
    else:
        # Return None if the pattern does not match
        return None


def extract_number1(input_string):
    # Define the regex pattern to match the desired part
    pattern = r'^_?[a-zA-Z]+(\d+)(_?\d+)?$'

    # Search for the pattern in the input string
    match = re.match(pattern, input_string)

    if match:
        # Return the first number part
        return match.group(1)
    else:
        # Return None if the pattern does not match
        return None


print(source('a1_2'))  # a
print(source_sample('a1_2'))  # a1
