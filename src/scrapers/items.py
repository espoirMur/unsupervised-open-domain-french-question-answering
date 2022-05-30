from scrapy.item import Item, Field
from src.utils.helpers import convert_date


class WebsiteItem(Item):
    """
    scraped data
    """
    def get_date(self, date, format): #pylint: disable=redefined-builtin, no-self-use
        """
        given a date local string and a format return a date object
        """
        return convert_date(date=date, format=format)

    title = Field()
    url = Field()
    summary = Field()
    content = Field()
    author = Field()
    website_origin = Field()
    posted_at = Field()
