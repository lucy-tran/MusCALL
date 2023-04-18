import subprocess
import json
import os
import csv

from muscall.utils.utils import get_root_dir

output_file_prefix = "Final"

def test(model_id, test_set_size, trial):
    command = "python3 evaluate.py %s retrieval %s_%s_%s %s" %(model_id, output_file_prefix, test_set_size, trial, test_set_size) 
    retrieval_metrics = subprocess.run(command, shell=True)

experiments_path = os.path.join("..", "save", "experiments")
experiments = os.listdir(experiments_path)
experiments.remove('get_best.py')
experiments.remove('records.tsv')

num_trials = 1

with open("results.tsv", "w", encoding='utf8', newline='') as record_file:
    tsv_writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')

    for exp in experiments:
        best_R10s = {}
        avg_R10s = {}
        test_set_size = 20
        best_R_10 = 0.0
        total_R_10 = 0.0
        best_trial = 1

        for trial in range(1, num_trials+1):
            test(exp, test_set_size, trial)

            output_file_path = os.path.join(experiments_path, exp, "%s_%s_%s.txt" %(output_file_prefix, test_set_size, trial))
            
            with open(output_file_path, 'r') as file:
                retrieval_metrics = json.loads(file.readline())

                trial_R_10 = retrieval_metrics["R@2"]
                total_R_10 += trial_R_10

                if trial_R_10 > best_R_10:
                    best_R_10 = trial_R_10
                    best_trial = trial

        best_R10s[test_set_size] = best_R_10
        avg_R10s[test_set_size] = total_R_10 / num_trials

        tsv_writer.writerow([exp, *best_R10s.values(), *avg_R10s.values()])

# best_R10s = {20: 1, 40: 2, 60: 3}
# print(*best_R10s.values())