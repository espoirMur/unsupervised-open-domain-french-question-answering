from scrapy.item import Item, Field

class WebsiteItem(Item):
    title = Field()
    url = Field()
    summary = Field()
    text_content = Field()
