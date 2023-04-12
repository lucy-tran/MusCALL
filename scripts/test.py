import subprocess
import json
import os
import csv

from muscall.utils.utils import get_root_dir

output_file_prefix = "Imagination"

def test(model_id, test_set_size, trial):
    command = "python3 evaluate.py %s retrieval %s_%s_%s %s" %(model_id, output_file_prefix, test_set_size, trial, test_set_size) 
    retrieval_metrics = subprocess.run(command, shell=True)

experiments_path = os.path.join("..", "save", "experiments")
experiments = os.listdir(experiments_path)
experiments.remove('get_best.py')
experiments.remove('records.tsv')
experiments.remove('2023-04-03-09_07_00')

with open("results.tsv", "w", encoding='utf8', newline='') as record_file:
    tsv_writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')
    tsv_writer.writerow(["experiment_id", "test_set_size", "best_R@10", "from trial#"])

    for exp in experiments:
        for test_set_size in range(20, 101, 20):
            best_R_10 = 0.0
            best_trial = 1

            for trial in range(1, 6):
                test(exp, test_set_size, trial)
                output_file_path = os.path.join(experiments_path, exp, "%s_%s_%s.txt" %(output_file_prefix, test_set_size, trial))
                
                with open(output_file_path, 'r') as file:
                    retrieval_metrics = json.loads(file.readline())

                    trial_R_10 = retrieval_metrics["R@10"]
                    print(trial_R_10)

                    if trial_R_10 > best_R_10:
                        best_R_10 = trial_R_10
                        best_trial = trial

            tsv_writer.writerow([exp, test_set_size, best_R_10, best_trial])
