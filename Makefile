MAIN = src/main.py
INIT = src/init_stock_data.py
DATA = shared/stock_data.pkl
SETTING = src/setting.yml

all: $(MAIN) $(DATA)
	python3 $<

$(DATA): $(INIT) $(SETTING)
	python3 $<

lint:
	black --check .
	isort --check .
	pflake8 .
	
format:
	isort .
	black .

.PHONY: lint format
