# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/itemt-pipeline.html


# useful for handling different item types with a single interface
from sqlalchemy.orm import sessionmaker
from src.scrapers.models import Article, engine


class SaveItemPipeline:
    """
    save scraped items to the database
    """
    def __init__(self):
        self.session = sessionmaker(bind=engine)

    def process_item(self, item, spider): #pylint: disable=unused-argument, missing-docstring
        session = self.session()
        if "title" in item:
            article = Article(**item)
            try:
                session.add(article)
                session.commit()
            except Exception: #pylint: disable=broad-except
                session.rollback()
            finally:
                session.close()

            return item
