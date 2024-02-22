import numpy as np
from datetime import datetime, timedelta

def is_leap_year(year: int):
    """
    Returns True if year is a leap year, False otherwise.
    
    Parameters:
    - year (int): The year to check.

    Returns:
    - bool: True if year is a leap year, False otherwise.
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def dictionary_to_array(dictionary: dict):
    """
    Converts a dictionary of lists into a single NumPy array.

    Parameters:
    - dictionary (dict): The dictionary to convert.

    Returns:
    - numpy.ndarray: A single array with the last dimension corresponding to the
                     dictionary's keys.
    """

    array = np.stack([dictionary[key] for key in dictionary.keys()], axis=-1)

    return array

def hour_to_datetime(hour: int):
    """
    Converts an hour value to a datetime object.

    Parameters:
    - hour (int): The hour to convert.

    Returns:
    - datetime.datetime: The corresponding datetime object.
    """
    base_date = datetime(1900, 1, 1, 0, 0, 0)
    result_date = base_date + timedelta(hours=hour)

    return result_date