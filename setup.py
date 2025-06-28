from setuptools import setup, find_packages
from typing import List

# Declaring variables for setup functions
PROJECT_NAME = "sensor-fault-detection"
VERSION = "0.0.1"
AUTHOR = "Rahul Shelke",
DESCRIPTION  = """The goal of this project is to provide a data-driven approach for early detection of Air Pressure System (APS) failures in Scania trucks, 
                  which can assist fleet operators and maintenance teams in preventing critical breakdowns, reducing downtime, and enhancing road safety."""
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements_list() -> List[str]:
    """
    Description : This function is going to return list of requirement
    mention in requirement.txt file
    return This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list

setup(
    name = PROJECT_NAME,
    version = VERSION,
    author = AUTHOR,
    description = DESCRIPTION,
    packages = find_packages(),
    install_requirements = get_requirements_list()
)