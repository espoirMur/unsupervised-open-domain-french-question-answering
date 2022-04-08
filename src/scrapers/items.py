from scrapy.item import Item, Field

class WebsiteItem(Item):
    title = Field()
    url = Field()
    summary = Field()
    content = Field()
    author = Field()
    website_origin = Field()
    posted_at = Field()