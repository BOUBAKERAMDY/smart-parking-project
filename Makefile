.PHONY: install train serve test lint docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python train.py

serve:
	uvicorn app:app --reload --port 8080

test:
	pytest tests/ -v

lint:
	ruff check .

docker-build:
	docker build -t smart-parking-api .

docker-run:
	docker run -p 8080:8080 smart-parking-api

clean:
	rm -rf models/*.keras models/*.pkl data/processed_occupancy.csv __pycache__ tests/__pycache__
