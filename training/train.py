from dataclasses import dataclass, field
import glob
import os
import subprocess
import yaml
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from utils import CustomTrainer, create_and_prepare_model, create_datasets


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    config: Optional[str] = field(
        default=None, metadata={"help": "The path to the config file"}
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="Salesforce/codegen25-7b-multi",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="wikimedia/wikipedia",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_path: Optional[str] = field(
        default="",
        metadata={"help": "The path to local dataset to use."},
    )
    dataset_subset: Optional[str] = field(
        default="", metadata={"help": "Subset of the dataset to use"}
    )
    dataset_num_entries: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of entries from the dataset to use. Set to -1 to use all entries."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    eval_steps: int = field(default=10, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    output_dir: str = field(
        default="outputs", metadata={"help": "Where to store the final model."}
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of dataset workers to use."}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tests things like proper saving/loading/logging of model"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded"},
    )
    use_pretrained: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use pretrained model"}
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "Whether to resume from checkpoint"}
    )
    save_total_limit: Optional[int] = field(
        default=5, metadata={"help": "Number of checkpoints to save"}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True, metadata={"help": "Whether to load best model at the end"}
    )


def main(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True

    # training arguments
    save_strategy = "steps"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # model
    model, tokenizer = create_and_prepare_model(args)

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    trainer = CustomTrainer(
        model,
        dataset_text_field=args.dataset_text_field,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.num_workers,
    )

    trainer.accelerator.print(f"{trainer.model}")

    # train
    last_checkpoint = None

    if (
        glob.glob(os.path.join(args.output_dir, "checkpoint-*/**"))
        and args.resume_from_checkpoint
    ):
        last_checkpoint = True
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    # Save everything else on main process
    if trainer.args.process_index == 0:
        print("Sharding model if >10GB...")
        # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
        # We run this in a subprocess to avoid interference from the accelerators.
        directory = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [
                "python",
                os.path.join(directory, "shard_checkpoint.py"),
                f"--output_dir={args.output_dir}",
            ],
            check=True,
        )
        if "training_args.bin" in os.listdir(args.output_dir):
            os.remove(os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)

    args = parser.parse_args_into_dataclasses()[0]
    if args.config:
        with open(args.config, "r") as f:
            config_args = yaml.safe_load(f)
        for k, v in config_args.items():
            setattr(args, k, v)

    main(args)
