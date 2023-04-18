import subprocess
import os

from muscall.utils.utils import get_root_dir

def train(learning_rate, batch_size):
    config_path = os.path.join("..", "configs", "tune", "lr%i" %learning_rate, "training_bs%i.yaml" %batch_size)
    print(config_path)
    command = "python3 train.py --config_path %s" %config_path
    subprocess.run(command, shell=True)


for lr in range(5, 7):
    for batch_size in range(3, 6):
        if (lr==5 and batch_size==3):
            for _ in range(2):
                train(lr, batch_size)
        else:
            for _ in range(3):
                train(lr, batch_size)

# for batch_size in range(3, 6):
#     for _ in range(10):
#         train(3, batch_size)


