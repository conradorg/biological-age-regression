from datetime import datetime
import subprocess
from airflow.decorators import dag, task
from airflow.providers.standard.operators.bash import BashOperator


@dag(start_date=datetime(2025, 1, 1), schedule=None, catchup=False)
def dag_hyperparam_tuning():

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /app && pipenv run python -u prepare_data.py",
    )

    # this version does not enable live logging:
    # @task.bash
    # def run_train():
    #     return "cd /app && pipenv run python -u train.py 2015 2017 0"
    # run_train()

    # this version enables live logging but is the older version:
    hyperparam_tuning = BashOperator(
        task_id="hyperparam_tuning",
        bash_command="echo $MLFLOW_TRACKING_URI && cd /app && pipenv run python -u train.py 2015 2017 0",
    )
    # return hyperparam_tuning

    prepare_data >> hyperparam_tuning


dag_hyperparam_instance = dag_hyperparam_tuning()


@dag(start_date=datetime(2025, 1, 1), schedule=None, catchup=False)
def dag_hyperparam_tuning_train_best():

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /app && pipenv run python -u prepare_data.py",
    )

    hyperparam_tuning = BashOperator(
        task_id="hyperparam_tuning",
        bash_command="cd /app && pipenv run python -u train.py 2015 2017 0",
    )

    train = BashOperator(
        task_id="run_train",
        bash_command="cd /app && pipenv run python -u train.py --hyperparam_tune False 2015 2017 0",
    )

    prepare_data >> hyperparam_tuning >> train  # define dependencies


dag_hyperparam_train_instance = dag_hyperparam_tuning_train_best()
