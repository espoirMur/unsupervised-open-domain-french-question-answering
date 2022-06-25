from datetime import datetime
from dateutil.parser import parse


def convert_date(date, format=None): #pylint: disable=redefined-builtin
    """
    given a date local string and a format return a date object
    """
    try:
        date = parse(date)
        if isinstance(date, datetime):
            return date
        formated_date = datetime.strptime(date, format)
        return formated_date
    except ValueError:
        formated_date = datetime.strptime(date, format)
        return formated_date
