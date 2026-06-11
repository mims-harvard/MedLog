.DEFAULT_GOAL := help

.PHONY: help
help: ##@ List available commands with their descriptions
	@printf "\nUsage: make <command>\n"
	@grep -F -h "##@" $(MAKEFILE_LIST) | grep -F -v grep -F | sed -e 's/\\$$//' | awk 'BEGIN {FS = ":*[[:space:]]*##@[[:space:]]*"}; \
	{ \
		if($$2 == "") \
			pass; \
		else if($$0 ~ /^#/) \
			printf "%s", $$2; \
		else if($$1 == "") \
			printf "     %-20s%s", "", $$2; \
		else \
			printf "    \033[34m%-20s\033[0m %s\n", $$1, $$2; \
	}'

.PHONY: install
install: ##@ Create the virtual environment and install the pre-commit hooks
	@echo "Creating virtual environment using uv..."
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ##@ Run code quality tools
	@echo "Checking lock file consistency with 'pyproject.toml'..."
	@uv lock --locked
	@echo "Linting code..."
	@uv run pre-commit run -a

.PHONY: clean
clean: ##@ Clean up caches and build artifacts
	@echo "Cleaning up the project..."
	@rm -rf `find . -name __pycache__`
	@rm -f `find . -type f -name '*.py[co]'`
	@rm -rf `find . -type d -name '.ipynb_checkpoints'`
	@rm -rf .cache .mypy_cache .pytest_cache .ruff_cache
