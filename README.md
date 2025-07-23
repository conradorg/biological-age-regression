# Phenoage Regression

This project demonstrates a MLOps service for estimating the **biological age** (PhenoAge) based on publicly available health data (NHANES).  
Unlike chronological age, biological age reflects a person's actual physiological condition and risk profile. It is increasingly used in preventive medicine, longevity research, and health technology. This project is done within the course MLOps Zoomcamp to demonstrate all visited MLOps concepts and best-practices.

The goal is to build a lightweight regression model that approximates **PhenoAge** — a clinically validated aging biomarker — using only **non-invasive, easily measurable features** such as BMI, blood pressure, physical activity, and lifestyle factors.

By replacing complex lab tests with accessible input data, the project shows how machine learning can support **low-threshold, scalable health insights**—and serves as a fully reproducible **MLOps case study** with potential applications in digital health tools.



## Setup (tested on Ubuntu/Linux)
### Virtual environment with Pipenv
`pipx` is used to install and run python tools in isolated virtual environments to not pollute the system python - in this case we use it for installing `pipenv`.
Install `pipx` into a dedicated Python virtual environment under your `.local/` directory to keep it isolated from system packages.
```bash
python3 -m venv ~/.local/pipx_venv
~/.local/pipx_venv/bin/pip install pipx
~/.local/pipx_venv/bin/pipx ensurepath  # add ~/.local/bin to PATH
ln -s ~/.local/pipx_venv/bin/pipx ~/.local/bin/pipx  # create symlink for pipx executable in ~/.local/bin/
exec $SHELL  # restart shell
pipx --version
```

`pipenv` is used to manage project dependencies and the virtual environment. Install it with `pipx`
```bash
pipx install pipenv

```

Inside the project repo:
```bash
pipenv --python "$(which python3)" install  # install pipenv with a specific python version (useful if you have multiple version of python installed)
pipenv lock
pipenv shell
```


Notes for first pipenv setup (not needed anymore):
```bash
pipenv install scikit-learn
```


## Dataset Description
The National Health and Nutrition Examination Survey (NHANES) collects health data of adults and children in the United States. 
- Overview: https://www.cdc.gov/nchs/nhanes/about/index.html
- Data Overview August 2021-August 2023: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
- Multiple topics: Demographics, Dietary Data, Examination Data, etc
    - for each topic you can find a dedicated page with 
        1. a Doc file explaining all variables and
        2. a Data file containing the collected data (XPT file)
    - e.g. Data Overview for Demographics: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2021-2023

    
## 0. Data Exploration Notebook



## TODOs documentation:
- explain NHANES dataset 
- explain preprocessing

## TODOs code:
- test data preparation (pandas merge)
