# Biological Age Regression

This project demonstrates a MLOps service for estimating the **biological age** (using the PhenoAge) based on publicly available health data (NHANES).  
Unlike chronological age, biological age reflects a person's actual physiological condition and risk profile. It is increasingly used in preventive medicine, longevity research, and health technology. This project is done within the course MLOps Zoomcamp to demonstrate all visited MLOps concepts and best-practices.

## Goal
The goal is to build a lightweight regression model that approximates **PhenoAge** — a clinically validated aging biomarker — using only **non-invasive, easily measurable features** such as BMI, blood pressure, physical activity, and lifestyle factors. 
The target variable PhenoAge is calculated from data which where obtained invasively. Unfortunately the data format and the data contained is changing partially from year to year in the public data, so I just used the years 2015 and 2017. One year serves for training, the other is divided in batches and one batch serves for validation.

By replacing complex lab tests with accessible input data, the project shows how machine learning can support **low-threshold, scalable health insights**—and serves as a fully reproducible **MLOps case study** with potential applications in digital health tools.

## Hint
At the moment the project implements
- ML training with a Support Vector Regression model using sklearn
- Experiment tracking: MLFlow (no model registry at the moment)
- Workflow orchestration: using Apache Airflow
- Model deployment: not deployed at the moment
- model monitoring: basic validation at the moment after training
- black: code formatter is used
- pre-commit hooks are used



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



## Dataset Description
The National Health and Nutrition Examination Survey (NHANES) collects health data of adults and children in the United States. 
- Overview: https://www.cdc.gov/nchs/nhanes/about/index.html
- Data Overview August 2021-August 2023: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
- Multiple topics: Demographics, Dietary Data, Examination Data, etc
    - for each topic you can find a dedicated page with 
        1. a Doc file explaining all variables and
        2. a Data file containing the collected data (XPT file)
    - e.g. Data Overview for Demographics: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2021-2023

    


## Usage
Hint: everything is executed locally at the moment.
- The project uses docker compose to boot up the developement environment with mlflow (experiments) and airflow (orchestration)
    - MLflow UI: http://localhost:5000
    - Airflow UI: http://localhost:8080
- **What is Airflow?** Apache Airflow is a tool for automating and scheduling workflows. In this project, it is used to define and run data and training pipelines as code (called DAGs - Directed Acyclic Graphs).
    - The (very simple) DAGs (workflows) are defined in the `airflow-dags/` folder.
    - Open the Airflow UI and trigger the DAGs manually as needed.
- Additional services in the docker compose file:
    - minio: used to provide an s3 bucket
    - createbucket: minio client which is used to create the s3 bucket
    - postgres_airflow: database for apache airflow orchestrator

Please copy the environment file examples `.env-docker.example` and `.env-local.example` to create  `.env-docker` and `.env-local`. And change the the environment variables if needed. You can start the docker containers with the following command:
```bash
sudo docker compose --env-file .env-docker up --build
```
please wait 20 seconds, then you can open mlflow and airflow to execute data loading, hyperparameter tuning und training tasks. Sometimes airflow workflows and tasks are not properly logged if the UI is used too early (you can have the web interface open but wait before you reload.) Sometimes the tasks always fail, in this case please restart docker compose again. (Ctrl + C and execute the command again)

When the UI is ready, you can on the left-hand side click on the DAG (workflow) section and click then on the "play/triangle" button. Then, a dialog window opens "Trigger DAG - dag_hyperparam_tuning_train_best" and you can just click on trigger. Afterwards, you can click on the latest run. The individual task runs are shown and you can inspect them and the logs.

You can also execute the code locally outside the docker container
```bash
pipenv run python src/prepare_data.py  # fetches the data, applies preprocessing and saves it to parquet format
pipenv run python src/train.py 2015 2017 0  # hyperparameter tuning (train year 2015, val year 2017, val batch 0)
pipenv run python src/train.py --hyperparam_tune False 2015 2017 0  # training the best three models from hyperparameter tuning
```



Useful commands:
```bash
sudo docker compose exec -it airflow bash  # enter a docker compose container (here: airflow container)
```
