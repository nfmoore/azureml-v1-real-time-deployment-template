import json
import sys
from argparse import ArgumentParser

from azureml.core import Dataset, Datastore, Workspace
from azureml.data.dataset_factory import DataType
from azureml.datadrift import DataDriftDetector

target_dataset_timestamp_column = "$aml_system_partition_date"


def parse_args(argv):
    ap = ArgumentParser("data_drift_setup")

    ap.add_argument("--subscription_id", required=True)
    ap.add_argument("--resource_group", required=True)
    ap.add_argument("--workspace_name", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--endpoint_name", required=True)
    ap.add_argument("--baseline_dataset_name", required=True)
    ap.add_argument("--data_drift_monitor_name", required=True)
    ap.add_argument("--compute_target", required=True)
    ap.add_argument("--feature_list", required=True)
    ap.add_argument("--frequency", default="Week")

    args, _ = ap.parse_known_args(argv)

    return args


def main():
    # Parse command line arguments
    args = parse_args(sys.argv[1:])

    # Retreive workspace
    workspace = Workspace.get(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        name=args.workspace_name,
    )

    # List data drift detectors
    drift_detector_list = DataDriftDetector.list(workspace)

    # Delete existing data drift detector
    for drift_monitor in drift_detector_list:
        if drift_monitor.name == args.data_drift_monitor_name:
            print("Deleteing existing data drift monitor...")
            drift_monitor.delete()

    # Retreive compute cluster
    compute_target = workspace.compute_targets[args.compute_target]

    # Grt model id and version
    model_name, model_version = args.model_id.split(":")

    # Define target dataset
    target_dataset_name = (
        f"inference-data-{model_name}-{model_version}-{args.endpoint_name}"
    )

    # Get current registered target dataset definition
    current_target_dataset = Dataset.get_by_name(workspace, name=target_dataset_name)
    current_target_dataset_definition = json.loads(current_target_dataset._definition)

    # Get current registered target dataset datasetore definition
    current_target_dataset_datastore_definition = current_target_dataset_definition[
        "blocks"
    ][0]["arguments"]["datastores"][0]

    # Define current registered target dataset datasetore
    target_dataset_datastore = Datastore(
        workspace, current_target_dataset_datastore_definition["datastoreName"]
    )

    # Define current registered target dataset datasetore path
    target_dataset_datastore_path = current_target_dataset_datastore_definition["path"]

    # Create updated target dataset with non-string feature data types
    target_dataset = Dataset.Tabular.from_delimited_files(
        path=(target_dataset_datastore, target_dataset_datastore_path),
        validate=False,
        set_column_types={
            "age": DataType.to_float(),
            "gender": DataType.to_string(),
            "height": DataType.to_float(),
            "weight": DataType.to_float(),
            "systolic": DataType.to_float(),
            "diastolic": DataType.to_float(),
            "cholesterol": DataType.to_string(),
            "glucose": DataType.to_string(),
            "smoker": DataType.to_string(),
            "alcoholic": DataType.to_string(),
            "active": DataType.to_string(),
            "cardiovascular_disease": DataType.to_string(),
        },
    )

    # Register updated dataset version
    target_dataset.register(
        workspace, name=target_dataset_name, create_new_version=True
    )

    # Assign timestamp column for Tabular Dataset to activate time series related APIs
    target_dataset = target_dataset.with_timestamp_columns(
        timestamp=target_dataset_timestamp_column
    )

    # Get baseline dataset
    baseline_dataset = Dataset.get_by_name(workspace, args.baseline_dataset_name)

    print("Variable [target_dataset]:", target_dataset)
    print("Variable [baseline_dataset]:", baseline_dataset)

    # Define features to monitor
    feature_list = args.feature_list.split(",")

    print("Variable [feature_list]:", args.feature_list)

    # Define data drift detector
    monitor = DataDriftDetector.create_from_datasets(
        workspace,
        args.data_drift_monitor_name,
        baseline_dataset,
        target_dataset,
        compute_target=compute_target,
        frequency=args.frequency,
        feature_list=feature_list,
    )

    print("Variable [monitor]:", monitor)

    # Enable the pipeline schedule for the data drift detector
    monitor.enable_schedule()


if __name__ == "__main__":
    main()
