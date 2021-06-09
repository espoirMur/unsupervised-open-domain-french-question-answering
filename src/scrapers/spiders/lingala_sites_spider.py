import scrapy
import lxml.html
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
        Rule(LinkExtractor(allow=r'.*/a($|/.*)'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        summary_xpath = '//div[@id="content"]/div/div[2]/div/div/div/div[1]/div[1]'
        body_xpath = '//*[@id="article-content"]/div[1]'
        title_xpath = '//*[@id="content"]/div/div[1]/div/div[2]/h1/text()'
        website_item = WebsiteItem()
        content_selector = Selector(response)
        title = content_selector.xpath(title_xpath).get()
        title = "-".join(title.split(" ")).strip()
        filename = self.create_filename(title)
        content_html = content_selector.xpath(body_xpath).get()
        summary_html = content_selector.xpath(summary_xpath).get()
        website_item["text_content"] = self.get_text_from_html(content_html)
        website_item['summary'] = self.get_text_from_html(summary_html)
        website_item["title"] = title
        with open(filename, 'w') as f:
            f.write(f"{website_item['summary']} \n {website_item['text_content']}")
        self.log(f'Saved file {filename}')

    def create_filename(self, title):
        base_folder = Path.cwd().parent.joinpath("data", 'raw', "lingala_articles", self.name)
        base_folder.mkdir(mode=0o777, parents=True, exist_ok=True)
        filename = base_folder.joinpath(title)
        return filename

    def get_text_from_html(self, html_content):
        """
        given a html content passed in parameter return the corresponding content

        Args:
            html_content ([type]): [description]
        """
        return lxml.html.fromstring(html_content).text_content().strip()
