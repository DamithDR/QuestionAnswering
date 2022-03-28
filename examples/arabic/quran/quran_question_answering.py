import json
import os
from os.path import exists

from examples.arabic.quran.quranqa22_eval import check_and_evaluate
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_training_set, format_n_diacritize_training_set, \
    format_n_diacritize_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import load_jsonl

import pyarabic.araby as araby


def run(
        learning_rate=4e-5,
        num_train_epochs=6,
        manual_seed=None,
        # model="aubmindlab/araelectra-base-generator",
        # model="aubmindlab/bert-base-arabertv2",
        # model="aubmindlab/bert-large-arabertv02",
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix",
        # model="bert-base-multilingual-cased",
        # model="bert-base-multilingual-uncased",
        diacritize=False,
        run_number=1
):
    raw_training_set_path = os.path.join(".", "data", "qrcd_v1.1_train.jsonl")
    training_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_train_formatted.jsonl")
    raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_dev.jsonl")
    dev_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_dev_formatted.jsonl")
    results_folder = os.path.join(".", "data", "run-files")
    file_no = len([name for name in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder,name))]) + 1
    results_file = os.path.join(".", "data", "run-files","assemble", "DTW_run" + str(run_number) + ".json")

    if not exists(training_set_path):
        if diacritize:
            format_n_diacritize_training_set(raw_training_set_path, training_set_path)
        else:
            format_training_set(raw_training_set_path, training_set_path)

    if not exists(dev_set_path):
        if diacritize:
            format_n_diacritize_dev_set(raw_dev_set_path, dev_set_path)
        else:
            format_dev_set(raw_dev_set_path, dev_set_path)

    model = QuestionAnsweringModel(
        "bert",
        model,
        args={"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "learning_rate": learning_rate,
              "manual_seed": manual_seed,
              "num_train_epochs": num_train_epochs},
    )

    # Train the model
    model.train_model(training_set_path)
    # result, text = model.eval_model(".\data\preprocess\output\qrcd_v1.1_dev_formatted.jsonl")
    # make predictions
    dev_set = load_jsonl(dev_set_path)

    answers, scores, ans_with_scores = model.predict(dev_set, n_best_size=5)
    results_dict = {}
    for result in ans_with_scores:
        ans_score_list = []
        rank = 1
        for i in range(0, len(result['answer'])):
            if len(result['answer'][i].strip()) != 0:
                record = {'answer': result['answer'][i], 'rank': rank, 'score': result['probability'][i]}
                ans_score_list.append(record)
                rank += 1
        results_dict[result['id']] = ans_score_list

    # dump_jsonl(results_dict, results_file)
    with open(results_file, "w", encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False)



if __name__ == '__main__':
    run()
