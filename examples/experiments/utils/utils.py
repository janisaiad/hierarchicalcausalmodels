
def store_dict_to_file(data_dict, filename):
    """
    Store a dictionary with variables in the format variablename_count or 
    variablename_count1_count2 in a text file.
    
    :param data_dict: Dictionary to store
    :param filename: Name of the file to store the dictionary in
    """
    with open(filename, 'w') as file:
        for key, value in data_dict.items():
            file.write(f"{key} -> {value}\n")


def load_dict_from_file(filename):
    """
    Load a dictionary from a text file in the format 'variablename_count -> value'
    or 'variablename_count1_count2 -> value'.
    
    :param filename: Name of the file to load the dictionary from
    :return: Dictionary with the loaded data
    """
    data_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(' -> ')
            try:
                # Attempt to convert value to int or float if possible
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    # If conversion fails, keep it as a string
                    pass
            data_dict[key] = value
    return data_dict

