import os
import csv

experiments = os.listdir()
experiments.remove('get_best.py')
experiments.remove('records.tsv')

with open("records.tsv", "w", encoding='utf8', newline='') as record_file:
   tsv_writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')

   for exp in experiments:
      train_log = os.path.join(exp, "train_log.tsv")
      best_metric = 0.0
      best_val_loss = 999.0

      with open(train_log, 'r') as file:
         # Read the lines
         rows = file.readlines()

         for row in rows:
            # Count the column count for the current line
            entries = row.split('\t')

            if len(entries) != 7 or entries[0] == 'Epoch':
               continue

            val_loss = float(entries[2])
            metric = float(entries[3])

            if val_loss < best_val_loss:
               best_val_loss = val_loss
            if metric > best_metric:
               best_metric = metric

      tsv_writer.writerow([exp, best_val_loss, best_metric])



         