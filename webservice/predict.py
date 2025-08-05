import os
import mlflow
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.svm import LinearSVR


client = mlflow.MlflowClient()
experiment_name = "best-n-SVR-models"
experiment = client.get_experiment_by_name(experiment_name)
if experiment is not None:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["metrics.val_mean_absolute_error ASC"], max_results=1
    )
else:
    raise ValueError(f"Experiment {experiment_name} does not exist.")

best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
feature_transform_uri = f"runs:/{best_run.info.run_id}/feature_transform"
target_transform_uri = f"runs:/{best_run.info.run_id}/target_transform"
model = mlflow.sklearn.load_model(model_uri)
feature_transform = mlflow.sklearn.load_model(feature_transform_uri)
target_transform = mlflow.sklearn.load_model(target_transform_uri)


def predict(features):
    preds = model.predict(features)
    return preds


app = Flask("phenoage-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    print("Received message!")
    inverse_ttf = lambda y: target_transform.inverse_transform(y.reshape((-1, 1)))

    input_data = request.get_json()
    print(request.get_json())
    print(type(input_data))
    input_data = pd.DataFrame([input_data])
    features = feature_transform.transform(input_data)
    pred = predict(features)
    pred_processed = inverse_ttf(pred)
    pred_processed = float(pred_processed[0])

    result = {"phenoage_prediction": pred_processed, "model_version": best_run.info.run_id}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
    print("Server started!")
