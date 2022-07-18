import json
from pathlib import Path
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from src.scrapers.spiders.base import BaseSpider


class ExetatCultGenSpider(BaseSpider):
    """
    scraper for http://exetat-rdc.com/index.php/culture-generale
    """
    name = 'exetat_cult_gen'
    start_urls = ['http://exetat-rdc.com/index.php/culture-generale']
    allowed_domains = ['exetat-rdc.com']
    website_origin='http://exetat-rdc.com'
    rules = (
        Rule(callback='callback', follow=True, link_extractor=LinkExtractor(allow=('index.php/culture-generale/[a-zA-Z0-9]'))),
    )

    def callback(self, response):
        """
        callback for each page
        """
        questions = response.css('.tqQuestion')
        for q in questions:
            question = q.css('span.tqHeader::text').get()
            correct_index = q.css('::attr(data-correct)').get()
            options = q.css('form label')
            options_array = []
            for option in options:
                options_text = option.css('::text').getall()
                for text in options_text:
                    tex = str(text).strip()
                    if tex != '':
                        options_array.append(tex)
            # print('question : ', question)
            # print('correct_index : ', correct_index)
            # print('options : ', options_array)
            # print('answer : ', options_array[int(correct_index)])

            # Insert data into a json file
            base_folder = Path.cwd().parent.joinpath(
                "data", "json", self.name
            )
            base_folder.mkdir(mode=0o777, parents=True, exist_ok=True)
            file_name = base_folder.joinpath(
                self.name + '.json'
            )
            with open(file_name, 'w') as f:
                json.dump({
                    'question': question,
                    'correct_index': correct_index,
                    'options': options_array,
                    'answer': options_array[int(correct_index)]
                }, f)
                f.write('\n')
                print('saved to : ', file_name)

            # with open('cult_gen.json', 'a') as file:
            #     file.write(json.dumps({
            #         'question': question,
            #         'options': options_array,
            #         'answer': options_array[int(correct_index)]
            #     }))
                # file.write('\n')
