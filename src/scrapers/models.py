from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.orm import declarative_base
from scrapy.utils.project import get_project_settings

Base = declarative_base()

def db_connect():
    engine = create_engine(get_project_settings().get("DB_CONNECTION_STRING"))
    return engine

def create_table(engine):
    Base.metadata.create_all(engine)

class Article(Base):
    __tablename__ = "article"

    id = Column(Integer, primary_key=True)
    title = Column(String(250), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    sumary = Column(Text)
    posted_at = Column(DateTime)
    website_origin = Column(String(250))
    url = Column(String(250))
    author = Column(String(250))