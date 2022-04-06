from src.scrapers.spiders.base import BaseSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from utils.helpers import default_parser

class PoliticoSpider(BaseSpider):
    name = 'politico'
    allowed_domains = ['politico.cd']
    start_urls = ['https://www.politico.cd']
    rules = (
        Rule(LinkExtractor(deny=r'.*rubrique+'), callback='callback', follow=False),
    )
    title_path = '.tdb-title-text::text'
    content_path = '.td-post-content * p::text'
    
    def callback(self, response):
        return default_parser(self, response, self.title_path, self.content_path, "french")