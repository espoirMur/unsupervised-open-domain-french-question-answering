from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class LaprosperiteSpyder(BaseSpider):
    name = 'laprosperite'
    start_urls = ['https://laprosperiteonline.net']
    allowed_domains = ['laprosperiteonline.net']
    website_origin='https://laprosperiteonline.net'
    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = 'h2.entry-title::text'
        content_path = 'section .entry-content p::text, section .entry-content p > strong::text, section .entry-content p > strong > em::text'
        author_path = '.entry-author > span::text'
        posted_at_path = '.entry-date > span::text'
        summary_path = '.entry-summary * p::text'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'author_path': author_path
        })
