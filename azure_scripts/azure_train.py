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
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core import ScriptRunConfig

import uuid
import argparse
from consts import (
    AZURE_DATASTORE,
    AZURE_ENV_VARIABLES,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_TENANT_ID,
    AZURE_WORKSPACE_NAME,
    LLM_OUT_NAME,
    CLUSTER_NAME_GPU,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Training")
    parser.add_argument(
        "--execution-type",
        type=str,
        choices=["single-node", "multi-node"],
        default="single-node",
        help="Type of execution (single-node or multi-node)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to use for multi-node execution",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--deepspeed", default=True, help="Use DeepSpeed for training")
    parser.add_argument(
        "--compute-target", default=CLUSTER_NAME_GPU, help="Compute target to use"
    )
    parser.add_argument(
        "--experiment-name", default="llm-training", help="Experiment name"
    )
    args = parser.parse_args()

    if args.deepspeed == False and args.execution_type == "multi-node":  # noqa: E712
        raise Exception("Multi-node execution requires DeepSpeed")

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

    docker_config = DockerConfiguration(use_docker=True, shm_size="128g")

    # create a new runconfig object
    runconfig = RunConfiguration()
    runconfig.environment_variables = AZURE_ENV_VARIABLES
    runconfig.environment = env
    runconfig.docker = docker_config
    runconfig.target = CLUSTER_NAME_GPU

    run_id = str(uuid.uuid4())
    output_dataset = (
        OutputFileDatasetConfig(destination=(datastore, f"llm/{run_id}"))
        .register_on_complete(name=LLM_OUT_NAME)
        .as_mount()
    )
    # input_folder = Dataset.get_by_name(
    #     ws, name=DATASET_OUT_NAME,
    # ).as_download()

    parent_directory = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.join(parent_directory, "..")

    node_count = args.num_nodes if args.execution_type == "multi-node" else 1
    process_count = node_count * args.num_gpus

    distr_config = PyTorchConfiguration(
        process_count=process_count, node_count=node_count
    )
    if args.deepspeed:
        command = [
            "accelerate launch",
            "--config_file training/configs/deepspeed_config.yaml",
        ]
    else:
        command = [
            "python",
        ]

    if args.execution_type == "multi-node":
        command.extend(
            [
                "--machine_rank $NODE_RANK",
                f"--num_machines {node_count}",
                f"--num_processes {process_count}",
                "--main_process_ip $MASTER_ADDR",
                "--main_process_port $MASTER_PORT",
            ]
        )
    command.extend(
        [
            "training/train.py",
            "--config training/configs/gpt2_train.yaml",
            # "--dataset_path",
            # input_folder,
            # "--dateset-name",
            # "wikipedia",
            "--output_dir",
            output_dataset,
            "--cache_dir",
            output_dataset,
        ]
    )
    src = ScriptRunConfig(
        source_directory=source_directory,
        command=command,
        compute_target=args.compute_target,
        environment=env,
        distributed_job_config=distr_config,
    )

    AZURE_ENV_VARIABLES["NCCL_DEBUG"] = "WARN"
    AZURE_ENV_VARIABLES["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    AZURE_ENV_VARIABLES["NCCL_SOCKET_IFNAME"] = "eth0"
    AZURE_ENV_VARIABLES["NCCL_IB_PCI_RELAXED_ORDERING"] = "1"
    AZURE_ENV_VARIABLES["UCX_TLS"] = "tcp"
    AZURE_ENV_VARIABLES["UCX_NET_DEVICES"] = "eth0"
    src.run_config.environment_variables = AZURE_ENV_VARIABLES
    src.run_config.docker = docker_config

    dataset_creation = CommandStep(
        name="LLM Training",
        outputs=[output_dataset],
        source_directory=source_directory,
        runconfig=src,
        allow_reuse=False,
    )

    exp = Experiment(workspace=ws, name="llm-training-multi-node")

    pipeline = Pipeline(ws, steps=dataset_creation)
    run = exp.submit(pipeline)
    description = args.experiment_name
    run.description = description
    run.display_name = description
    run.wait_for_completion(show_output=True)
