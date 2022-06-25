
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_database_session(database_url):
    """
    Create a database session for database task

    Args:
        credentials (dict): credentials to use to connect to the db

    Returns:
        [tuple]: database session and the engine
    """
    engine = create_engine(database_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session, engine
