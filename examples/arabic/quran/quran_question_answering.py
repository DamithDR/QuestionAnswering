import json
import os
from os.path import exists

from examples.arabic.quran.quranqa22_eval import check_and_evaluate
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_training_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import load_jsonl


def run(
        learning_rate=4e-5,
        num_train_epochs=1,
        manual_seed=None,
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix"
):
    raw_training_set_path = os.path.join(".", "data", "qrcd_v1.1_train.jsonl")
    training_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_train_formatted.jsonl")
    raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_dev.jsonl")
    dev_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_dev_formatted.jsonl")
    results_file = os.path.join(".", "data","run-files", "DTW_run01.json")

    if not exists(training_set_path):
        format_training_set(raw_training_set_path, training_set_path)

    if not exists(dev_set_path):
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

    answers, scores, ans_with_scores = model.predict(dev_set, n_best_size=3)
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

    check_and_evaluate(run_file=results_file, gold_answers_file=raw_dev_set_path)

    return answers


if __name__ == '__main__':
    run()
