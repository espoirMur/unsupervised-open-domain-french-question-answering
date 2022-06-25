from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class LepotentielSpider(BaseSpider): # pylint: disable=abstract-method
    """
    scraper for https://lepotentiel.cd
    """
    name = 'lepotentiel'
    website_origin = 'https://lepotentiel.cd'
    start_urls = ['https://lepotentiel.cd']
    allowed_domains = ['lepotentiel.cd']

    rules = (
        Rule(callback='callback', follow=True),
    )

    def callback(self, response):
        """
        callback for each page
        """
        title_path = 'h1.elementor-heading-title::text, h1.elementor-heading-title > strong::text'
        content_path = '.elementor-widget-container p::text, .elementor-widget-container p > strong::text, .elementor-widget-container p > strong > em::text'
        author_path = '.elementor-post-info__item--type-author::text'
        posted_at_path = '.entry-date > span::text'
        summary_path = '.entry-summary * p::text'
        return self.default_parser(response, css_paths={
            'title_path': title_path,
            'content_path': content_path,
            'summary_path': summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        })
