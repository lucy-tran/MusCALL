import os
import argparse
from omegaconf import OmegaConf
import numpy as np

from muscall.tasks.retrieval import Retrieval
from muscall.tasks.classification import Zeroshot
from muscall.utils.utils import get_root_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MusCALL model")

    parser.add_argument(
        "model_id",
        type=str,
        help="experiment id under which trained model was saved",
    )
    parser.add_argument(
        "task",
        type=str,
        help="name of the evaluation task (retrieval or zeroshot)",
    )

    parser.add_argument(
        "output_file_name",
        type=str,
        default="output.txt"
    )

    parser.add_argument(
        "test_set_size",
        type=int,
        help="size of the random testing set",
        default=1000,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of dataset for zeroshot classification",
    )
    parser.add_argument(
        "--device_num",
        type=str,
        default="0",
    )


    args = parser.parse_args()

    return args


def main():
    params = parse_args()
    model_id = params.model_id

    muscall_config = OmegaConf.load(
        os.path.join(get_root_dir(), "save/experiments/{}/config.yaml".format(model_id))
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    output_file_name = params.output_file_name
    
    if params.task == "retrieval":
        evaluation = Retrieval(muscall_config, params.test_set_size)
        txt_file = os.path.join(get_root_dir(), "save/experiments/{}/{}.txt".format(model_id, output_file_name))
    elif params.task == "zeroshot":
        evaluation = Zeroshot(muscall_config, params.dataset_name)
        txt_file = os.path.join(get_root_dir(), "save/experiments/{}/{}.txt".format(model_id, output_file_name))
    else:
        raise ValueError("{} task not supported".format(params.task))

    retrieval_metrics = evaluation.evaluate(txt_file)
    return retrieval_metrics

main()
