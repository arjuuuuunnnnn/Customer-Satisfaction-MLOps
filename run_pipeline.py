from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="//data/olist_customers_dataset.csv")

# now run this pipeline, and then you'll get something in the form pasted below and paste the same thing below and run
# the same thing, and then you are done
# mlflow ui --backend-store-uri-file:/home/hemanth/only_machine_learning
