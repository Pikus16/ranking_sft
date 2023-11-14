from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, MistralForCausalLM, BitsAndBytesConfig
import logging
from typing import Union, Optional, Dict, List
import click
import torch
from marco_dataset import MarcoDataset
from train_utils import seed_all, RESPONSE_TEMPLATE, RESPONSE_TEMPLATE_TOKENS
import pandas as pd
import numpy as np
import os
from datasets import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_INPUT_MODEL = "mistralai/Mistral-7B-v0.1"

# def get_dataset(csv_path: str, train_size: float = 0.6, val_size:float = 0.2, test_size: float = 0.2, seed: int = 42) -> Dict: 
#     seed_all(seed)
#     assert train_size + test_size + val_size == 1.0
    
#     val_size = train_size + val_size
#     df = pd.read_csv(csv_path)
#     train_df, validate_df, test_df = np.split(df.sample(frac=1, random_state=seed), 
#                        [int(train_size*len(df)), int(val_size*len(df))])
#     train_dataset = MarcoDataset(train_df)
#     val_dataset = MarcoDataset(validate_df)
#     test_dataset = MarcoDataset(test_df)
#     return {
#         "train" :train_dataset, "val" :val_dataset, "test" : test_dataset
#     }
def get_dataset(csv_base_path: str):
    train_dataset = load_dataset('csv', data_files=os.path.join(csv_base_path,'train.csv'))['train']
    val_dataset = load_dataset('csv', data_files=os.path.join(csv_base_path,'val.csv'))['train']
    test_dataset = load_dataset('csv', data_files=os.path.join(csv_base_path,'test.csv'))['train']
    return {
        "train" :train_dataset, "val" :val_dataset, "test" : test_dataset
    }
    

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_model_and_tokenizer(model_name: str, gradient_checkpointing: bool = False, load_in_8bit=False, load_in_4bit=True):
    logger.info(f"Loading model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load_in_8bit or load_in_4bit:
        logger.info('Using quantization')
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
            if False
            else {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
        
    model = MistralForCausalLM.from_pretrained(model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype
    )  
    print_trainable_parameters(model)                      
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train(
        dataset_path: str,
        local_output_dir: str,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        #bf16: bool = False,
        lr: float = 1e-5,
        epochs: int = 3,
        gradient_checkpointing: bool = True,
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 400,
        warmup_steps: Optional[int] = 100,
        model_name: str = DEFAULT_INPUT_MODEL,
        #test_size: Union[float, int] = 1000
    ):
    if not os.path.exists(local_output_dir):
        os.mkdir(local_output_dir)

    model, tokenizer = get_model_and_tokenizer(model_name=model_name)
    split_dataset = get_dataset(dataset_path)
    data_collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE_TOKENS, tokenizer=tokenizer)    

    # enable fp16 if not bf16
    #fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        #fp16=fp16,
        #bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        #deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        #save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        #disable_tqdm=True,
        remove_unused_columns=False,
        #local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            #"gate_proj",
            #"up_proj",
            #"down_proj",
            #"lm_head",
        ],
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["val"],
        args=training_args,
        data_collator=data_collator,
        peft_config=lora_config,
        packing=False,
        dataset_text_field='text',
        max_seq_length=4096
    )
    trainer.save_model(output_dir='test')
    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--model-name", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dataset-path", type=str, help="CSV to load in", required=True)
@click.option("--epochs", type=int, default=2, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
#@click.option(
#    "--test-size", type=int, default=1000, help="Number of test records for evaluation, or ratio of test records."
#)
@click.option("--warmup-steps", type=int, default=100, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
#@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
#@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
#@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
# @click.option(
#     "--local_rank",
#     type=str,
#     default=True,
#     help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
# )
#@click.option("--bf16", type=bool, default=None, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise