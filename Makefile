.PHONY: list docs

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'

init:
	poetry install --sync --with dev
	pre-commit install

style:
	pre-commit run black -a

lint:
	pre-commit run ruff -a

mypy:
	dmypy run auto_reports

test:
	python -m pytest -vlx

# clean_notebooks:
# 	pre-commit run nbstripout -a

# exec_notebooks:
# 	python -m nbconvert --to notebook --execute  --ExecutePreprocessor.kernel_name=python3 --stdout examples/* >/dev/null

# docs:
# 	make -C docs html

deps:
	pre-commit run poetry-lock -a
	pre-commit run poetry-check -a
	pre-commit run poetry-export -a
