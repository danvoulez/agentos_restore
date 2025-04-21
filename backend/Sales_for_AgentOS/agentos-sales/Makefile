PY=python -m
ENV?=.env

dev:
	ENV_FILE=$(ENV) uvicorn app.main:app --reload --port 8000

worker:
	ENV_FILE=$(ENV) celery -A app.worker.celery_app worker -Q sales -l INFO -P eventlet

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app

test:
	pytest -q

coverage:
	pytest --cov=app --cov-report=term-missing