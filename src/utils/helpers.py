from scrapy import Selector

def default_parser(self, response, title_css_path, content_css_path, language="lingala"):
        selector = Selector(response)
        try:
            title = selector.css(title_css_path).get()
            content = selector.css(content_css_path).getall()
            title = '-'.join(title.split(' ')).strip()
            content = ''.join(content)
            filename = self.create_filename(title, language)
            with open(filename, 'w') as f:
                f.write(content)
            self.log(f'Saved file {filename}')
        except Exception as e:
            print(f'Error while parsing {response.url} : \n', e.__str__())
            pass