# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/itemt-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from src.scrapers.models import Article, create_table, db_connect
from sqlalchemy.orm import sessionmaker


class SaveItemPipeline:
    def __init__(self):
        engine = db_connect()
        create_table(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        session = self.Session()
        article = Article(**item)
        try:
            session.add(article)
            session.commit()
            self.logger.info(f"done adding the post  to the database")
        except Exception:
            session.rollback()
        finally:
            session.close()

        return item
