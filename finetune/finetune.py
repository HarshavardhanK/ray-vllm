
import ray
ray.init()

#Define parameters
task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 4
use_gpu = True
num_workers = 1


from datasets import load_dataset
actual_task = "mnli" if task == "mnli-mm" else task
datasets = load_dataset("glue", actual_task)

#Convert to Ray Data
import ray.data
ray_datasets = {
    "train": ray.data.from_huggingface(datasets["train"]),
    "validation": ray.data.from_huggingface(datasets["validation"]),
    "test": ray.data.from_huggingface(datasets["test"]),
}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#Define task to keys mapping
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

#Create preprocessing function
import torch
import numpy as np
from typing import Dict

def collate_fn(examples: Dict[str, np.array]):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        outputs = tokenizer(
            list(examples[sentence1_key]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
    else:
        outputs = tokenizer(
            list(examples[sentence1_key]),
            list(examples[sentence2_key]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

    outputs["labels"] = torch.LongTensor(examples["label"])

    # Move all input tensors to GPU
    for key, value in outputs.items():
        outputs[key] = value.cuda()

    return outputs

#Setup for training
import torch
import numpy as np
from evaluate import load as load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
metric_name = (
    "pearson"
    if task == "stsb"
    else "matthews_correlation"
    if task == "cola"
    else "accuracy"
)
model_name = model_checkpoint.split("/")[-1]
validation_key = (
    "validation_mismatched"
    if task == "mnli-mm"
    else "validation_matched"
    if task == "mnli"
    else "validation"
)
name = f"{model_name}-finetuned-{task}"

#Calculate the maximum steps per epoch
max_steps_per_epoch = ray_datasets["train"].count() // (batch_size * num_workers)

#Define training function
def train_func(config):
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    metric = load_metric("glue", actual_task)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("eval")

    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=collate_fn
    )
    eval_ds_iterable = eval_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=collate_fn
    )

    print("max_steps_per_epoch: ", max_steps_per_epoch)

    args = TrainingArguments(
        output_dir=name,
        eval_strategy="epoch",
        save_strategy="no",  #Disable saving checkpoints
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 2),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        max_steps=max_steps_per_epoch * config.get("epochs", 2),
        disable_tqdm=True,  #declutter the output a little
        no_cuda=not use_gpu,  #you need to explicitly set no_cuda if you want CPUs
        report_to="none",
        #Add memory efficient training options
        gradient_accumulation_steps=4,  #Accumulate gradients before updating
        fp16=True,  #Use mixed precision training
        gradient_checkpointing=True,  #Enable gradient checkpointing
        optim="adamw_torch_fused",  #Use fused AdamW optimizer
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    print("Starting training")
    trainer.train()

#Create the trainer
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    datasets={
        "train": ray_datasets["train"],
        "eval": ray_datasets["validation"],
    },
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,  # Keep only one checkpoint
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
            checkpoint_frequency=0,  #Only save at the end
        ),
    ),
)

#Start training
result = trainer.fit()

#Hyperparameter tuning with Ray Tune
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers.async_hyperband import ASHAScheduler

tune_epochs = 4
tuner = Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "learning_rate": tune.grid_search([2e-5, 2e-4, 2e-3, 2e-2]),
            "epochs": tune_epochs,
        }
    },
    tune_config=tune.TuneConfig(
        metric="eval_loss",
        mode="min",
        num_samples=1,
        scheduler=ASHAScheduler(
            max_t=tune_epochs,
        ),
    ),
    run_config=RunConfig(
        name="tune_transformers",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
)

tune_results = tuner.fit()

#Get best result and view results
best_result = tune_results.get_best_result()
tune_results.get_dataframe().sort_values("eval_loss")

#Load and push the model to Hugging Face Hub (optional)
import os
from ray.train import Checkpoint

checkpoint: Checkpoint = best_result.checkpoint

with checkpoint.as_directory() as checkpoint_dir:
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# To share the model with the community (uncomment to use)
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub()