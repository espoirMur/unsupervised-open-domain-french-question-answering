from scrapy.item import Item, Field
from itemloaders.processors import MapCompose, TakeFirst

class WebsiteItem(Item):
    title = Field()
    url = Field()
    summary = Field()
    text_content = Field()

class PoliticoItem(Item):
    title = Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    content = Field(
        output_processor=TakeFirst()
    )
    img_url = Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    posted_at = Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    author = Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )