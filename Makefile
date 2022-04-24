# Makefile to initialize the project
SHELL := /bin/bash

.ONESHELL:

.PHONY: init

init:
	python3 -m venv env
	source env/bin/activate
	pip3 install -e .
	pip3 install -r requirements.txt

clean:
	@rm -r env
	@rm -r *.egg-info

# for this call to work do "make params=<command_line_arguments_for_program> train" 	
train:
	source env/bin/activate
	python3 torch_template/main.py $(params)
