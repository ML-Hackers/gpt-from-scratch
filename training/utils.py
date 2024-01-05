from datasets import load_dataset
from tqdm import tqdm
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from trl import SFTTrainer
import numpy as np


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
            num_proc=args.num_workers,
        )
    if args.dataset_num_entries > 0:
        dataset.select(range(args.dataset_num_entries))
    dataset = dataset["train"].train_test_split(test_size=0.01)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = train_data
    valid_dataset = valid_data

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    if args.use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            use_flash_attention_2=args.use_flash_attn,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            token=os.environ.get("HF_API_TOKEN", None),
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            torch_dtype="auto",
            token=os.environ.get("HF_API_TOKEN", None),
            use_bfloat16=args.bf16,
        )
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, use_flash_attention_2=args.use_flash_attn
        )
        print("Model config", model.dtype)

    def print_num_params(model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {(num_params / 1e9):.2f}B")

    print_num_params(model)
    model.init_weights()

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_API_TOKEN", None),
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


class CustomTrainer(SFTTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        TRAINER_STATE_NAME = "trainer_state.json"

        if self.args.should_save:
            self.state.save_to_json(
                os.path.join(staging_output_dir, TRAINER_STATE_NAME)
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()
        # Then go through the rewriting process starting on process 0

        if staging_output_dir != output_dir:
            with self.args.main_process_first(
                desc="Renaming model checkpoint folder to true location",
                local=self.args.save_on_each_node,
            ):
                if os.path.exists(staging_output_dir):
                    try:
                        os.rename(staging_output_dir, output_dir)
                    except Exception:
                        print("Error renaming checkpoint directory, skipping")
                    pass

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            try:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
            except Exception:
                print("Error rotating checkpoints, skipping")
                pass
