import csv
import numpy as np

from quran_question_answering import run

manual_seed = [None, 3000]
models = ["CAMeL-Lab/bert-base-arabic-camelbert-mix",
          "CAMeL-Lab/bert-base-arabic-camelbert-ca",
          "aubmindlab/bert-base-arabertv2"]
header = ['Learning Rate', "No of epochs", 'Seed', "Results", "Text"]
seed = None
if __name__ == '__main__':
    for model in models:
        output_file = "outputs\\" + model.split('/')[1] + "-permutations.csv"
        with open(output_file, 'w', encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for l_rate in np.arange(0.00001, 0.0001, 0.00001):
                for no_epochs in range(1, 5, 1):
                    results, text = run(learning_rate=l_rate,
                                        num_train_epochs=no_epochs,
                                        manual_seed=seed,
                                        model=model)
                    data = [l_rate, no_epochs, seed, results, text]
                    writer.writerow(data)
