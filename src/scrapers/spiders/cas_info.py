from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class CasInfoSpider(BaseSpider):
    website_origin = 'https://cas-info.ca'
    start_urls = ['https://cas-info.ca']
    allowed_domains = ['cas-info.ca']
    name = 'cas_info'

    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = 'h1.title > a::text'
        content_path = 'article p::text, article p > strong::text, article p > strong > em::text'
        author_path = '.mg-info-author-block .media-body .media-heading > a::text'
        posted_at_path = '.entry-date > span::text'
        summary_path = '.entry-summary * p::text'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'author_path': author_path
        })
