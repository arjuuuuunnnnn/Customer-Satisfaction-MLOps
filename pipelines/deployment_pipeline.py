import json
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from pipelines.utils import get_data_for_test
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(required_integration=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data


@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
):
    '''
    Implements a simple model deployment trigger that looks at the input model accuracy and decides if it is good enough
    to deploy or not
    '''
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name: str,
        running: bool = True,
        model_name: str = "model",
) -> MLFlowDeploymentService:
    # get the mlflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline: {pipeline_name}"
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline is for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(
     service: MLFlowDeploymentService,
        data: str,
) -> np.array:

    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_length",
        "product_description_length",
        "products_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json.list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
        data_path: str,
        min_accuracy: float = 0,
        workers: int = 1,
        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction