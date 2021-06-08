import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from src.scrapers.items import WebsiteItem
from pathlib import Path
from scrapy.selector import Selector



class LingalaSiteScraper(CrawlSpider):
    name = 'voa_lingala'
    allowed_domains = ['voalingala.com']
    start_urls = ['https://www.voalingala.com/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        summary_xpath = '//div[@id="content"]/div/div[2]/div/div/div/div[1]/div[1]/text()'
        body_xpath = '//div[@id="article-content"/text()]'
        title = response.css('title::text').get()
        filename = self.create_filename(title)
        website_item = WebsiteItem()
        content_selector = Selector(response)
        website_item["text_content"] = content_selector.xpath(body_xpath).extract()[0]
        website_item['summary'] = content_selector.xpath(summary_xpath).extract()[0]
        # feat need to fix, scrapping path
        with open(filename, 'wb') as f:
            f.write(website_item.text_content)
        self.log(f'Saved file {filename}')

    def create_filename(self, title):
        base_folder = Path.cwd().parent.joinpath("data", 'raw', "lingala_articles", self.name)
        base_folder.mkdir(mode=0o777, parents=True, exist_ok=True)
        title = title.replace(" ", "-")
        filename = base_folder.joinpath(title)
