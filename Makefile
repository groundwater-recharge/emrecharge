.PHONY: clean

clean:
	rm -rf build
	rm -rf dist

test:
	pytest

lint:
	pylint emrecharge

format:
	isort emrecharge
	black emrecharge

build:
	python -m build

deploy-check:
	python -m twine check dist/*

deploy-test:
	python -m twine upload --repository-url=https://test.pypi.org/legacy/ dist/*

deploy: deploy-check
	python -m twine upload dist/*
