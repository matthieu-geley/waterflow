import mlflow
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI to local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Instantiate MLflow client
client = MlflowClient()

# Create or get experiment
experiment_name = "experiment_water_quality"
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Start a new run in the experiment
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Save meta-data as tags or params
    mlflow.set_tag("project", "waterflow")
    mlflow.set_tag("description", "Experiment for water quality analysis")
    mlflow.log_param("initiated_by", "x_mat")
    # Add more meta-data as needed

    print(f"Started run with ID: {run.info.run_id}")