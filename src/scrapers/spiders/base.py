from pathlib import Path

import lxml.html
from scrapy import Selector
from scrapy.spiders import CrawlSpider
from src.scrapers.items import WebsiteItem


class BaseSpider(CrawlSpider):
    def create_filename(self, title, language="lingala"):
        base_folder = Path.cwd().parent.joinpath(
            "data", "raw", f"{language}_articles", self.name
        )
        base_folder.mkdir(mode=0o777, parents=True, exist_ok=True)
        filename = base_folder.joinpath(title)
        return filename

    def get_text_from_html(self, html_content):
        """
        given a html content passed in parameter return the corresponding content

        Args:
            html_content ([type]): [description]
        """
        return lxml.html.fromstring(html_content).text_content().strip()

    def default_parser(self, response, css_paths={}, date_format=None):
        selector = Selector(response)
        try:
            title_css_path = css_paths['title_path']
            content_css_path = css_paths['content_path']
            sumary_path = css_paths['sumary_path']
            posted_at_path = css_paths['posted_at_path']
            author_path = css_paths['author_path']

            title = selector.css(title_css_path).get()
            content = selector.css(content_css_path).getall()
            author = selector.css(author_path).get()
            sumary = selector.css(sumary_path).get()
            posted_at = selector.css(posted_at_path).get()
            content = ''.join(content)
            if(title and content):
                website_item = WebsiteItem(
                    title=title,
                    content=content,
                    url=response.url,
                    website_origin=self.start_urls[0],
                    author=author,
                    sumary=sumary
                )
                if posted_at:
                    date = website_item.get_date(posted_at, date_format)
                    website_item['posted_at'] = date
                    
                yield website_item
        except Exception as e:
            print(f'Error while parsing {response.url} : \n', e.__str__())
