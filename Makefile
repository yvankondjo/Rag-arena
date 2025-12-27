.PHONY: help download index benchmark test parallel report clean

help:
	@echo "RAGBench-12x Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install dependencies"
	@echo ""
	@echo "Data & Indexing:"
	@echo "  make download         - Download BEIR SciFact dataset"
	@echo "  make index            - Build indexes (doc-level)"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark        - Run full 12-config benchmark (sequential)"
	@echo "  make parallel         - Run 12-config benchmark (3 parallel)"
	@echo "  make test             - Quick smoke test (10 queries)"
	@echo ""
	@echo "Analysis:"
	@echo "  make report           - Generate report.md"
	@echo "  make clean            - Clean results/"
	@echo ""

install:
	uv sync

download:
	python -m src.cli download --dataset scifact

index:
	python -m src.cli index --dataset scifact

benchmark:
	python -m src.cli benchmark --dataset scifact

parallel:
	python -m src.cli benchmark --dataset scifact --parallel

test:
	python -m pytest tests/test_smoke_run.py -v

test-dry:
	python -m src.run_experiment --dataset scifact --dry-run --max-configs 12

report:
	python -m src.cli report --results-dir results --output results/report.md

clean:
	rm -rf results/runs/*
	rm -rf results/*.csv
	rm -rf results/*.md

# Quick start: download + index + test
quickstart: download index test
	@echo "✓ Quickstart complete!"
