from scrapy import Selector

from scrapers.items import WebsiteItem

def default_parser(self, response, language="lingala", css_paths={}):
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
            content = ''.join(content)
            print('orign : \n', self.start_urls[0])
            if(title and content):
                website_item = WebsiteItem()
                website_item["title"] = title
                website_item["content"] = content
                website_item["url"] = response.url
                website_item["website_origin"] = self.start_urls[0]
                website_item['author'] = author
                yield website_item
        except Exception as e:
            print(f'Error while parsing {response.url} : \n', e.__str__())