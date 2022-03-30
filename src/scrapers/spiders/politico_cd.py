import scrapy
from scrapy.loader import ItemLoader

from src.scrapers.items import PoliticoItem

class PoliticosSpider(scrapy.Spider):
    name = 'politico_cd'

    start_urls = ['https://www.politico.cd']

    def parse(self, response):
        self.logger.info("scraping politicos")

        headline_link = response.css('.entry-title a::attr(href)').get()
        loader = ItemLoader(item=PoliticoItem(), response=response)
        loader.add_css('img_url', '.td-module-thumb a span::attr(data-img-url)')
        item = loader.load_item()
        yield scrapy.Request(headline_link, self.parse_headline, meta={ 'item': item })

    def parse_headline(self, response):
        self.logger.info("Parsing headline")
        item = response.meta['item']
        loader = ItemLoader(item=item, response=response)
        loader.add_css('title', '.tdb-title-text::text')
        loader.add_css('author', '.tdb-author-name::text')
        loader.add_css('posted_at', '.td-module-date::attr(datetime)')
        content = response.css('.td-post-content * p::text').getall()
        loader.add_value('content', ''.join(content))

        yield loader.load_item()