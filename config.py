import os
from dotenv import load_dotenv
from urllib.parse import quote as urlquote

load_dotenv()

POSTGRES_USER = os.environ.get('POSTGRES_USER', '')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')
POSTGRES_HOST = os.environ.get('POSTGRES_HOST', '')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '')
POSTGRES_DB = os.environ.get('POSTGRES_DB', '')
DATABASE_URL = "postgresql://{}:{}@{}:{}".format(POSTGRES_USER,
                                                 urlquote(POSTGRES_PASSWORD),
                                                 POSTGRES_HOST,
                                                 POSTGRES_PORT)
