import csv
import os

import numpy as np

from examples.arabic.quran.assembler import assemble_results
from examples.arabic.quran.quranqa22_eval import check_and_evaluate
from quran_question_answering import compute

manual_seed = [None, 3000]
models = ["CAMeL-Lab/bert-base-arabic-camelbert-mix",
          "CAMeL-Lab/bert-base-arabic-camelbert-ca",
          "aubmindlab/bert-base-arabertv2"]
header = ['Learning Rate', "No of epochs", 'Seed', "Results", "Text"]
seed = None
# if __name__ == '__main__':
#     for model in models:
#         output_file = "outputs\\" + model.split('/')[1] + "-permutations.csv"
#         with open(output_file, 'w', encoding="utf-8", newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(header)
#             for l_rate in np.arange(0.00001, 0.0001, 0.00001):
#                 for no_epochs in range(1, 5, 1):
#                     results, text = run(learning_rate=l_rate,
#                                         num_train_epochs=no_epochs,
#                                         manual_seed=seed,
#                                         model=model)
#                     data = [l_rate, no_epochs, seed, results, text]
#                     writer.writerow(data)

if __name__ == '__main__':
    results_file = os.path.join(".", "data", "run-files", "DTW_01.json")
    raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_dev.jsonl")
    for i in range(1, 10):
        compute(run_number=i)

    assemble_results()
    check_and_evaluate(run_file=results_file, gold_answers_file=raw_dev_set_path)
