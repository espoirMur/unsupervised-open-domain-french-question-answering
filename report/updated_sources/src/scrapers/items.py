from itemloaders.processors import MapCompose, TakeFirst
from scrapy.item import Field, Item


class WebsiteItem(Item):
    title = Field()
    url = Field()
    summary = Field()
    text_content = Field()


class PoliticoItem(Item):
    title = Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
    content = Field(output_processor=TakeFirst())
    posted_at = Field(
        input_processor=MapCompose(str.strip), output_processor=TakeFirst()
    )
    author = Field(input_processor=MapCompose(str.strip), output_processor=TakeFirst())
