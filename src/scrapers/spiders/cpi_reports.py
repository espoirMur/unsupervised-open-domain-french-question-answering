import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from src.scrapers.items import WebsiteItem
from src.scrapers.spiders.base import BaseSpider
from scrapy.selector import Selector



class CpiReportScraper(BaseSpider):
    name = 'cpi_reports'
    # https://www.icc-cpi.int//Pages/item.aspx?name=180613-OTP-stat&ln=lingala
    #https://www.icc-cpi.int/test-new-master/Pages/pr-new.aspx?name=170331-otp-stat&ln=Lingala
    allowed_domains = ['icc-cpi.int']
    start_urls = ['https://www.icc-cpi.int']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )


    def parse_item(self, response):
        le = LinkExtractor()
        for link in le.extract_links(response):
            print(link.text, 10 * '==*')

