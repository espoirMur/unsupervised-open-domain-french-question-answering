from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from src.scrapers.spiders.base import BaseSpider


class MediacongoSpider(BaseSpider):
    name = "mediacongo"
    allowed_domains = ["mediacongo.net"]
    start_urls = ["https://www.mediacongo.net"]
    website_origin = "https://www.mediacongo.net"
    rules = (
        Rule(LinkExtractor(allow=r'.article-actualite*', allow_domains=('mediacongo.net')), callback='callback', follow=True),
    )

    def callback(self, response):
        title_path = '.first_article > h1::text'
        content_path = '.first_article_text > p::text, .first_article_text > p > strong::text'
        author_path='.one_article_who strong span::text'
        posted_at_path='.first_article > span::text'
        summary_path='.entry-content * p::text'
        date_format = '%Y-%m-%d'
        return self.default_parser(response, css_paths={
            'title_path':title_path,
            'content_path':content_path,
            'summary_path':summary_path,
            'posted_at_path': posted_at_path,
            'author_path': author_path
        }, date_format=date_format)
