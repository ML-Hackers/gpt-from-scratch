from dataclasses import dataclass, field
import os
import subprocess
import yaml
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from trl import SFTTrainer

from utils import SaveDeepSpeedModelCallback, create_and_prepare_model, create_datasets


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
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
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


def main(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    # training arguments
    is_deepspeed_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"
    )
    save_strategy = "no" if is_deepspeed_enabled else "steps"
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
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
        # ddp_timeout=7200,
    )

    # model
    model, tokenizer = create_and_prepare_model(args)

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    trainer = SFTTrainer(
        model,
        dataset_text_field=args.dataset_text_field,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.num_workers,
    )

    # trainer = Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    trainer.accelerator.print(f"{trainer.model}")

    if is_deepspeed_enabled:
        trainer.add_callback(
            SaveDeepSpeedModelCallback(trainer, save_steps=args.save_steps)
        )

    # train
    trainer.train()

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if is_deepspeed_enabled:
        # trainer.accelerator.wait_for_everyone()
        # state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
        # unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        # if trainer.accelerator.is_main_process:
        # unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        trainer.save_model(args.output_dir)
        # trainer.accelerator.wait_for_everyone()
    else:
        if args.push_to_hub:
            trainer.push_to_hub()
        else:
            trainer.save_model(args.output_dir)
            # Save the tokenizer to args.output_dir
            tokenizer.save_pretrained(args.output_dir)

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
