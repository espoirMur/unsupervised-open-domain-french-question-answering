from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from scrapy.spiders import Rule
from src.scrapers.items import WebsiteItem
from src.scrapers.spiders.base import BaseSpider


class KarismaTVScraper(BaseSpider): #pylint: disable=abstract-method

    """
    scraper for https://www.karismatv.cd
    """

    name = "karisma_tv"
    allowed_domains = ["karismatv.cd"]
    start_urls = ["https://www.karismatv.cd/"]
    website_origin = "https://www.karismatv.cd"

    rules = (
        Rule(LinkExtractor(allow=r".*id=\d+$"), callback="parse_item", follow=True),
    )

    def parse_item(self, response):
        
        """
            parsing content from response
        """

        body_xpath = "/html/body/div[1]/div/div/div[1]/div/div/div[2]/div/div[2]"
        title_xpath = "/html/body/div[1]/div/div/div[1]/div/div/div[1]/a[1]/h6/text()"
        author_path = '.post-author > a::text'
        website_item = WebsiteItem()
        content_selector = Selector(response)
        title = content_selector.xpath(title_xpath).get()
        author = content_selector.css(author_path).get()
        content_html = content_selector.xpath(body_xpath).get()
        website_item["content"] = self.get_text_from_html(content_html)
        website_item["title"] = title
        website_item["website_origin"] = self.website_origin
        website_item["author"] = author
        website_item["url"] = response.url
