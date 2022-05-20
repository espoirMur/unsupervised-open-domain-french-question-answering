from scrapy.item import Item, Field
from src.utils.helpers import convert_date


class WebsiteItem(Item):
    def get_date(self, date, format):
        return convert_date(date=date, format=format)

    title = Field()
    url = Field()
    summary = Field()
    content = Field()
    author = Field()
    website_origin = Field()
    posted_at = Field()
