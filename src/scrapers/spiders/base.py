from pathlib import Path
import lxml.html
from scrapy.spiders import CrawlSpider

class BaseSpider(CrawlSpider):
    def create_filename(self, title):
        base_folder = Path.cwd().parent.joinpath("data", 'raw', "lingala_articles", self.name)
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
