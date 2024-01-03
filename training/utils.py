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


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


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
            args.dataset_name,
            use_auth_token=False,
            num_proc=args.num_workers
        )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     infinite=True,
    #     seq_length=args.max_seq_length,
    #     chars_per_token=chars_per_token,
    #     content_field=args.dataset_text_field,
    #     shuffle=True,
    #     add_eos_token=False,
    # )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     valid_data,
    #     infinite=False,
    #     seq_length=args.max_seq_length,
    #     chars_per_token=chars_per_token,
    #     content_field=args.dataset_text_field,
    #     shuffle=False,
    #     add_eos_token=False,
    # )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    bnb_config = None
    load_in_8bit = args.use_8bit_quantization

    if args.use_4bit_quantization:
        print("USING 4BIT QUANTIZATION")
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if args.use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            use_flash_attention_2=args.use_flash_attn,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            token=os.environ.get("HF_API_TOKEN", None)
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name,
            load_in_8bit=load_in_8bit,
            quantization_config=bnb_config,
            use_flash_attention_2=args.use_flash_attn,
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
