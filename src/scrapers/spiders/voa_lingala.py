import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from scrapy.spiders import Rule
from src.scrapers.items import WebsiteItem
from src.scrapers.spiders.base import BaseSpider


class VoaLingalaScraper(BaseSpider):
    name = "voa_lingala"
    allowed_domains = ["voalingala.com"]
    start_urls = ["https://www.voalingala.com/"]
    website_origin = "https://www.voalingala.com/"

    rules = (
        Rule(LinkExtractor(allow=r".*/a($|/.*)"), callback="parse_item", follow=True),
    )

    def parse_item(self, response):
        content_path = '#article-content > div > p::text'
        title_path = '.title.pg-title::text'
        date_path = '.published .date time::attr(datetime)'
        author_path = '.links__item-link::text'
        content_selector = Selector(response)
        title = content_selector.css(title_path).get()
        date = content_selector.css(date_path).get()
        author = content_selector.css(author_path).get()
        content = content_selector.css(content_path).getall()
        if(title and content):
            website_item = WebsiteItem()
            valid_date = website_item.get_date(date, None)
            website_item["content"] = content
            website_item["title"] = title
            website_item["posted_at"] = valid_date
            website_item["author"] = author
            website_item["url"] = response.url
            website_item["website_origin"] = self.website_origin

            yield website_item
        
