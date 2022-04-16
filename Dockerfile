FROM python:3.7
COPY ./src src
COPY pyproject.toml .
COPY poetry.lock .

ENV PYTHONPATH="${PYTHONPATH}:${CWD}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.13 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
        software-properties-common \
        curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python \
        poetry install --no-dev  # respects

CMD [ "poetry", "run", "scrapy", "runspider", "./src/run_spiders.py" ]