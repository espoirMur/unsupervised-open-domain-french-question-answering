from datetime import datetime
from dateutil.parser import parse


def convert_date(date, format=None):
    try:
        date = parse(date)
        if type(date) == datetime:
            return date
        else:
            formated_date = datetime.strptime(date, format)
            return formated_date
    except ValueError: # Not sure about this one @jacob , you can check.. 
        formated_date = datetime.strptime(date, format)
        return formated_date
