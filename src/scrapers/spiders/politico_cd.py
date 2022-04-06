from src.scrapers.spiders.base import BaseSpider
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from src.scrapers.items import PoliticoItem

class PoliticoSpider(BaseSpider):
    name = 'politico'
    allowed_domains = ['politico.cd']
    start_urls = ['https://www.politico.cd']

    rules = (
        Rule(LinkExtractor(deny=r'.*rubrique+'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        title_path = '.tdb-title-text::text'
        content_path = '.td-post-content * p::text'
        selector = Selector(response)
        title = selector.css(title_path).get()
        content = selector.css(content_path).getall()
        title = '-'.join(title.split(' ')).strip()
        content = ''.join(content)
        politico_item = PoliticoItem()
        politico_item['title'] = title
        politico_item['content'] = content
        filename = self.create_filename(title)
        with open(filename, 'w') as f:
            f.write(content)
        self.log(f'Saved file {filename}')