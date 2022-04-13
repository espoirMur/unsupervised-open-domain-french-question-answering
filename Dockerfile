FROM python:3.7
ADD . .

RUN pip install -r requirements.txt

CMD [ "scrapy", "runspider", "./src/run_spiders.py" ]