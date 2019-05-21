init:
	conda env create -f environment.yml

offline:
	python offline.py

online:
	python server.py
