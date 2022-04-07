from src.scrapers.spiders.base import BaseSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from utils.helpers import default_parser

class SevenPer7Spider(BaseSpider):
    name='7sur7'
    allowed_domains = ['7sur7.cd']
    start_urls = ['https://www.7sur7.cd']
    rules = (
        Rule(LinkExtractor(allow_domains=('7sur7.cd')), callback='callback', follow=True),
    )

    title_path = 'h1.page-header span::text'
    content_path = '.content * p::text, .content * p > em::text, .content h3 > strong::text, .content * p::text'
    
    def callback(self, response):
        return default_parser(self, response, self.title_path, self.content_path, "french")