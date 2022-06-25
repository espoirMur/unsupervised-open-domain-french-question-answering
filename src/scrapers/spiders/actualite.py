from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
from src.scrapers.spiders.base import BaseSpider


class ActualiteSpider(BaseSpider):
    name = "actualite"
    allowed_domains = ["actualite.cd"]
    start_urls = ["https://www.actualite.cd"]
    website_origin = 'https://www.actualite.cd'
    
    rules = (
        Rule(LinkExtractor(allow_domains=('actualite.cd')), callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = '.views-field-title > span::text'
        content_path = '.views-field .views-field-body .field-content > p::text, .field-content > blockquote > p::text'
        author_path='.field-content > p > strong::text'
        posted_at_path='.first_article > span::text'
        summary_path='.entry-content * p::text'
        return self.default_parser(response, css_paths={
            'title_path':title_path,
            'content_path':content_path,
            'summary_path':summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        })
