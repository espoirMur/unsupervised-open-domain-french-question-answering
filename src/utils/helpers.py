from datetime import datetime
from dateutil.parser import parse
from dateutil.parser._parser import ParserError


def convert_date(date, format=None):
    try:
        date = parse(date)
        if type(date) == datetime:
            return date
        else:
            formatedDate = datetime.strptime(date, format)
            return formatedDate
    except ParserError:
        formatedDate = datetime.strptime(date, format)
        return formatedDate