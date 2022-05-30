from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor

from src.scrapers.spiders.base import BaseSpider

class SevenPer7Spider(BaseSpider): # pylint: disable=abstract-method
    """
    scraper for https://www.7sur7.cd
    """
    name = '7sur7'
    allowed_domains = ['7sur7.cd']
    start_urls = ['https://www.7sur7.cd']
    website_origin = 'https://www.7sur7.cd'
    
    rules = (
        Rule(LinkExtractor(allow_domains=('7sur7.cd')), callback='callback', follow=True),
    )
    
    def callback(self, response):
        """
        callback for each page
        """
        title_path = 'h1.page-header span::text'
        content_path = '.content * p::text, .content * p > em::text, .content h3 > strong::text, .content * p::text'
        summary_path = '.summary::text'
        posted_at_path = '.post_date::text'
        author_path = '.content * p > strong::text'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        })
