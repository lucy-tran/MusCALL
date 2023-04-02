import subprocess
import os

from muscall.utils.utils import get_root_dir

def train(learning_rate, batch_size):
    config_path = os.path.join("..", "configs", "tune", "lr%i" %learning_rate, "training_bs%i.yaml" %batch_size)
    print(config_path)
    command = "python3 train.py --config_path %s" %config_path
    subprocess.run(command, shell=True)


for lr in range(4, 7):
    for batch_size in range(3, 9):
        if (batch_size==3 and lr==4):
            for i in range(8):
                train(lr, batch_size)
        if (batch_size==3 and lr==5) or (batch_size==4 and lr==5):
            for i in range(9):
                train(lr, batch_size)
        else:
            for i in range(10):
                train(lr, batch_size)

