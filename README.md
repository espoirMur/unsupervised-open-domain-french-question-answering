multilingual-drc-news-chatbot


This project is a chatbot project that will be answering question about Congolese news.

This project have inside it many scrapers, and in case of running each of these scrapers, you should proceed as shown below :

-First, you need to setup a venv and activate it
-Make sure you install all dependecies inside the requirements.txt file with **pip install -r requirements.txt**
-And then you can run the scraper you want with **scrapy crawl 'scraper_name'**

After running the scraper and it succefully finished scraping the website, for each page of this website the scraper will save the file whose name is the title of the page and will have as content the content of the web page in data directory