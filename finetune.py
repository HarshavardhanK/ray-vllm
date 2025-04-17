from pprint import pprint
import ray

ray.init()

pprint(ray.cluster_resources())

use_gpu = True
num_workers = 1

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

from datasets import load_dataset

actual_task = "mnli" if task == "mnli-mm" else task
datasets = load_dataset("glue", actual_task)

