import wandb
from wandb.sdk.wandb_run import Run


def get_run() -> Run:
    return wandb.init(project="jax-fashion-mnist", entity="dl_hierarchy")
