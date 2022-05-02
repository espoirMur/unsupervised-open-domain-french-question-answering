from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class ForumDesasSpider(BaseSpider):
    name = 'forumdesas'
    start_urls = ['https://forumdesas.net/']
    allowed_domains = ['forumdesas.net']
    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = 'h1.elementor-heading-title::text'
        content_path = '.elementor-widget-container p.has-text-align-justify::text, .elementor-widget-container p.has-text-align-justify > strong::text'
        author_path = '.elementor-widget-container p.has-text-align-justify:last-of-type > strong::text'
        posted_at_path = '.elementor-post-info__item--type-date::text'
        sumary_path = '.entry-summary * p::text'
        date_format = '%d/%m/%Y'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'sumary_path': sumary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        }, date_format=date_format)