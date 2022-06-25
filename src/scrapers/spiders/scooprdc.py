from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class ScooprdcSpider(BaseSpider): # pylint: disable=abstract-method
    """
    scraper for https://scooprdc.net
    """
    name = 'scooprdc'
    website_origin = 'https://scooprdc.net'
    start_urls = ['https://scooprdc.net/']
    allowed_domains = ['scooprdc.net']

    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        """
        callback for each page
        """
        title_path = '.tdb-title-text::text'
        content_path = '.td-post-content * p::text'
        author_path = '.tdb-author-name::text'
        posted_at_path = '.td-module-date::attr(datetime)'
        summary_path = '.summary'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        })
