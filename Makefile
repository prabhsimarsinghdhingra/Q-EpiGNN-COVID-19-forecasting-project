.PHONY: install data preprocess test lint smoke all clean
install:
	pip install -r requirements-dev.txt
data:
	./download_data.sh
preprocess:
	python src/preprocess.py
test:
	pytest --cov=src --cov-report=term-missing
lint:
	ruff check src tests
smoke:
	python src/train.py --models persistence ar_ridge lstm qepignn --seeds 0 --epochs 2 --out ci_smoke.json
all:
	./run_all.sh
clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml junit.xml results/ci_smoke.json
