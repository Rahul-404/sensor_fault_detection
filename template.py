import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)

project_name = "sensor_fault_detection"

list_of_files = [
    ".github/workflows/main.yaml",
    "config/model.yaml",
    "config/schema.yaml",
    "docs/.gitkeep",
    "flowcharts/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/constant/training_pipeline/__init__.py",
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/ml/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    "notebooks/.gitkeep",
    "requirements.txt",
    "setup.py",
    ".dockerignore",
    "docker-compose.yaml",
    "Dockerfile",
    "templates/index.html",
    "static/css/style.css",
    "app.py",
    "demo.py"
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
