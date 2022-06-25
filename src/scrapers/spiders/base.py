from pathlib import Path

import lxml.html
from scrapy import Selector
from scrapy.spiders import CrawlSpider
from src.scrapers.items import WebsiteItem


class BaseSpider(CrawlSpider): #pylint: disable=abstract-method
    """
    Base spider class for our scrapers
    """

    def create_filename(self, title, language="lingala"):
        """
        given a title return a path to the file
        """
        base_folder = Path.cwd().parent.joinpath(
            "data", "raw", f"{language}_articles", self.name
        )
        base_folder.mkdir(mode=0o777, parents=True, exist_ok=True)
        filename = base_folder.joinpath(title)
        return filename

    @classmethod
    def get_text_from_html(cls, html_content):
        """
        given a html content passed in parameter return the corresponding content

        Args:
            html_content ([type]): [description]
        """
        return lxml.html.fromstring(html_content).text_content().strip()

    def default_parser(self, response, css_paths={}, date_format=None): #pylint: disable=dangerous-default-value
        """
        default parser for a website giving css paths and date format
        """
        selector = Selector(response)
        try:
            title_css_path = css_paths['title_path']
            content_css_path = css_paths['content_path']
            summary_path = css_paths['summary_path']
            posted_at_path = css_paths.get('posted_at_path', 'null')
            author_path = css_paths['author_path']

            title = selector.css(title_css_path).get()
            content = selector.css(content_css_path).getall()
            author = selector.css(author_path).get()
            summary = selector.css(summary_path).get()
            posted_at = selector.css(posted_at_path).get()
            content = ''.join(content)
            if(title and content):
                website_item = WebsiteItem(
                    title=title,
                    content=content,
                    url=response.url,
                    website_origin=self.website_origin, #pylint: disable=no-member
                    author=author,
                    summary=summary
                )
                if posted_at:
                    date = website_item.get_date(posted_at, date_format)
                    website_item['posted_at'] = date
                    
        except Exception as error: #pylint: disable=broad-except
            self.logger.error(f'Error while parsing {response.url} : \n', error.__str__())
