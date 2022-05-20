from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class Actu30Spider(BaseSpider):
    name = 'actu30'
    website_origin = 'https://actu30.cd'
    start_urls = ['https://actu30.cd']
    allowed_domains = ['actu30.cd']

    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = 'h1.tdb-title-text::text'
        content_path = '.td-post-content * p::text'
        author_path = '.tdb-author-name::text'
        posted_at_path = '.td-module-date::attr(datetime)'
        summary_path = '.sumary'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        })
