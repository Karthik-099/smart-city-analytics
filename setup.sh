#!/bin/bash
pip install -r requirements.txt
git init
dvc init
echo "data" >> .gitignore
echo "mlruns" >> .gitignore
echo "*.pyc" >> .gitignore
git add .
git commit -m "Initial project setup"
echo "Run 'dvc repro' to execute pipeline"
echo "Download dataset to data/raw/traffic_data.csv"