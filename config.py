import os
from dotenv import load_dotenv
from urllib.parse import quote as urlquote

load_dotenv()

POSTGRES_USER = os.environ.get('POSTGRES_USER', '')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')
POSTGRES_HOST = os.environ.get('POSTGRES_HOST', '')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', 5432)
POSTGRES_DB = os.environ.get('POSTGRES_DB', '')
DATABASE_URL = "postgresql://{}:{}@{}:{}/{}".format(POSTGRES_USER,
                                                    urlquote(POSTGRES_PASSWORD),
                                                    POSTGRES_HOST,
                                                    POSTGRES_PORT,
                                                    POSTGRES_DB)

PRODIGY_LOCAL_DB_NAME = os.environ.get('PRODIGY_LOCAL_DB_NAME', '')
PRODIGY_LOCAL_DB_USER = os.environ.get('PRODIGY_LOCAL_DB_USER', '')
PRODIGY_DATABASE_URL = "postgresql://{}:{}@{}:{}/{}".format(PRODIGY_LOCAL_DB_USER,
                                                            urlquote(POSTGRES_PASSWORD),
                                                            "localhost",
                                                            POSTGRES_PORT,
                                                            PRODIGY_LOCAL_DB_NAME)
