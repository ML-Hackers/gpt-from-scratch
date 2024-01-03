"""
Run LLM Training
"""
import os
from azureml.core import (
    Workspace,
    Datastore,
    Experiment,
    Environment,
)
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import RunConfiguration, DockerConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import CommandStep
from azureml.data import OutputFileDatasetConfig
from azureml.core import Dataset
from consts import (
    AZURE_DATASTORE,
    AZURE_ENV_VARIABLES,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_TENANT_ID,
    AZURE_WORKSPACE_NAME,
    DATASET_OUT_NAME,
    LLM_OUT_NAME,
    CLUSTER_NAME_GPU,
)


if __name__ == "__main__":
    interactive_auth = InteractiveLoginAuthentication(tenant_id=AZURE_TENANT_ID())

    ws = Workspace(
        subscription_id=AZURE_SUBSCRIPTION_ID(),
        resource_group=AZURE_RESOURCE_GROUP(),
        workspace_name=AZURE_WORKSPACE_NAME(),
        auth=interactive_auth,
    )
    datastore = Datastore.get(ws, AZURE_DATASTORE)

    env = Environment.from_conda_specification(
        "llm_finetuning", "training/training.yml"
    )
    # env.docker.base_image = None
    # env.docker.base_dockerfile = 'Dockerfile.triton'
    # env.python.user_managed_dependencies = True
    docker_config = DockerConfiguration(use_docker=True, shm_size="128g")

    # create a new runconfig object
    runconfig = RunConfiguration()
    runconfig.environment_variables = AZURE_ENV_VARIABLES
    runconfig.environment = env
    runconfig.docker = docker_config
    runconfig.target = CLUSTER_NAME_GPU

    output_dataset = (
        OutputFileDatasetConfig(
            destination=(datastore, "llm/{run-id}")
        )
        .register_on_complete(name=LLM_OUT_NAME)
        .as_mount()
    )
    # input_folder = Dataset.get_by_name(
    #     ws, name=DATASET_OUT_NAME,
    # ).as_download()
    
    parent_directory = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.join(parent_directory, "..")

    dataset_creation = CommandStep(
        name="LLM Training",
        command=[
            # "accelerate launch",
            # "--config_file training/configs/deepspeed_config.yaml",
            "python",
            "training/train.py",
            "--config training/configs/gpt_train.yaml",
            # "--dataset_path",
            # input_folder,
            "--dateset-name",
            "wikipedia",
            "--output_dir",
            output_dataset,
            "--cache_dir",
            output_dataset,
        ],
        outputs=[output_dataset],
        
        source_directory=source_directory,
        runconfig=runconfig,
        allow_reuse=False,
    )

    exp = Experiment(workspace=ws, name="llm-training")

    pipeline = Pipeline(ws, steps=dataset_creation)
    run = exp.submit(pipeline)
    description = "LLM training. Deepspeed"
    run.description = description
    run.display_name = description
    run.wait_for_completion(show_output=True)
