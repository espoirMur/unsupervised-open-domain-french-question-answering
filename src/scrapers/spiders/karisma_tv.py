import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from src.scrapers.items import WebsiteItem
from src.scrapers.spiders.base import BaseSpider
from scrapy.selector import Selector



class KarismaTVScraper(BaseSpider):
    name = 'karisma_tv'
    allowed_domains = ['karismatv.cd']
    start_urls = ['https://www.karismatv.cd/']

    rules = (
        Rule(LinkExtractor(allow=r'.*id=\d+$'), callback='parse_item', follow=True),
    )


    def parse_item(self, response):
        body_xpath = '/html/body/div[1]/div/div/div[1]/div/div/div[2]/div/div[2]'
        title_xpath = '/html/body/div[1]/div/div/div[1]/div/div/div[1]/a[1]/h6/text()'
        website_item = WebsiteItem()
        content_selector = Selector(response)
        title = content_selector.xpath(title_xpath).get()
        title = "-".join(title.split(" ")).strip()
        filename = self.create_filename(title)
        content_html = content_selector.xpath(body_xpath).get()
        website_item["text_content"] = self.get_text_from_html(content_html)
        website_item["title"] = title
        with open(filename, 'w') as f:
            f.write(website_item['text_content'])
        self.log(f'Saved file {filename}')

