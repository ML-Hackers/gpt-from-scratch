import random
import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig, 
    AutoModelForCausalLM
)


class SaveDeepSpeedModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            # self.trainer.accelerator.wait_for_everyone()
            # state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            # unwrapped_model = self.trainer.accelerator.unwrap_model(
            #     self.trainer.deepspeed
            # )
            if self.trainer.accelerator.is_main_process:
                print("Saving model checkpoint to:", args.output_dir)
                # unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.save_model(
                os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            )
            # self.trainer.accelerator.wait_for_everyone()
        return control


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens

def load_from_files(dataset_path: str):
    """
    Load dataset from files.
    """
    data_files = {
        "train": os.path.join(dataset_path, "train_dataset.csv"),
        "test": os.path.join(dataset_path, "validation_dataset.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)

    return dataset


def create_datasets(tokenizer, args):
    if args.dataset_path:
        dataset = load_from_files(args.dataset_path)
    else:
        dataset = load_dataset(
            path=args.dataset_name,
            name=args.dataset_subset,
            use_auth_token=False,
            num_proc=args.num_workers
        )
    if (args.dataset_num_entries > 0):
        dataset.select(range(args.dataset_num_entries))
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def create_and_prepare_model(args):

    
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     load_in_8bit=load_in_8bit,
    #     quantization_config=bnb_config,
    #     # use_flash_attention_2=args.use_flash_attn,
    #     # device_map=device_map,
    #     # use_cache=not args.use_gradient_checkpointing,
    #     trust_remote_code=True,
    #     cache_dir=args.cache_dir,
    #     token=os.environ.get("HF_API_TOKEN", None)
    # )

    config = AutoConfig.from_pretrained(
        args.model_name,
        # use_flash_attention_2=args.use_flash_attn,
        # device_map=device_map,
        # use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        token=os.environ.get("HF_API_TOKEN", None)
    )
    model = AutoModelForCausalLM.from_config(config)
    def print_num_params(model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {(num_params / 1e9):.2f}B")

    print_num_params(model)
    model.init_weights()
    
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True,
                                              token=os.environ.get("HF_API_TOKEN", None))
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
