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
        website_item["summary"] = self.get_text_from_html(summary_html)
        website_item["title"] = title
        with open(filename, "w") as f:
            f.write(f"{website_item['summary']} \n {website_item['text_content']}")
        self.log(f"Saved file {filename}")
