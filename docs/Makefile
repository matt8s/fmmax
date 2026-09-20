# Install npm and pip dependencies
install:
	@echo "Installing npm and pip dependencies..."
	npm ci
	python -m pip install docstring_parser nbformat nbconvert

# Check if required packages are installed
check-packages:
	@echo "Checking if required packages are installed..."
	@python -c "import docstring_parser, nbformat, nbconvert"
	@which npm > /dev/null || (echo "Error: npm not installed. Please install npm."; exit 1)

# Run gen_API_from_docstrings.py
gen-api:
	@echo "Running gen_API_from_docstrings.py..."
	python gen_API_from_docstrings.py

# Run gen_tutorials_from_notebooks.py
gen-tutorials:
	@echo "Running gen_tutorials_from_notebooks.py..."
	python gen_tutorials_from_notebooks.py

# Start the docusaurus server
start:
	@echo "Starting the application..."
	npm run start

# Run everything
all: check-packages gen-api gen-tutorials start

# Remove all nonessential files
clean:
	@echo "Removing all files described in .gitignore..."
	git clean -Xdf

deploy: install check-packages gen-api gen-tutorials
	npm run build
