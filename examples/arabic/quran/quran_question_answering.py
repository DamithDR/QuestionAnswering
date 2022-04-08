import argparse
import json
import os
from os.path import exists

from pyarabic import araby

from examples.arabic.quran.assembler import assemble_results
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_training_set, format_n_diacritize_training_set, \
    format_n_diacritize_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import load_jsonl
from examples.arabic.quran.quranqa22_eval import check_and_evaluate
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel


def compute(
        learning_rate=4e-5,
        num_train_epochs=6,
        manual_seed=None,
        # model="aubmindlab/araelectra-base-generator",
        # model="aubmindlab/bert-base-arabertv2",
        # model="aubmindlab/bert-large-arabertv02",
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix",
        # model="bert-base-multilingual-cased",
        # model="bert-base-multilingual-uncased",
        model_type="bert",
        diacritize=False,
        run_number=1,
        output_path=None
):
    raw_training_set_path = os.path.join(".", "data", "qrcd_v1.1_train.jsonl")
    training_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_train_formatted.jsonl")
    raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_dev.jsonl")
    # raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_test_noAnswers.jsonl")
    dev_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_dev_formatted.jsonl")
    # dev_set_path = os.path.join(".", "data", "preprocess", "output", "qrcd_v1.1_test_noAnswers_formatted.jsonl")
    results_folder = os.path.join(".", "data", "run-files")
    file_no = len(
        [name for name in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder, name))]) + 1
    if output_path is None:
        results_file = os.path.join(".", "data", "run-files", "assemble",
                                    model.replace('/', '-') + str(run_number) + ".json")
    else:
        results_file = os.path.join(".", "data", "run-files", "assemble", output_path)

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
        # "bert",
        model_type=model_type,
        model_name=model,
        args={"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "learning_rate": learning_rate,
              "manual_seed": manual_seed,
              "num_train_epochs": num_train_epochs},
    )

    # Train the model
    model.train_model(training_set_path)
    # make predictions
    dev_set = load_jsonl(dev_set_path)

    answers, scores, ans_with_scores = model.predict(dev_set, n_best_size=5)
    results_dict = {}
    for result in ans_with_scores:
        ans_score_list = []
        rank = 1
        for i in range(0, len(result['answer'])):
            if len(result['answer'][i].strip()) != 0:
                if diacritize:
                    record = {'answer': araby.strip_diacritics(result['answer'][i]), 'rank': rank,
                              'score': result['probability'][i]}
                else:
                    record = {'answer': result['answer'][i], 'rank': rank, 'score': result['probability'][i]}
                ans_score_list.append(record)
                rank += 1
        results_dict[result['id']] = ans_score_list

    # dump_jsonl(results_dict, results_file)
    with open(results_file, "w", encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False)
    return results_file


def run(models=[]):
    raw_dev_set_path = os.path.join(".", "data", "qrcd_v1.1_dev.jsonl")
    local_model_path = os.path.join(".", "models", "transferlearn",
                                    "CAMeL-Lab-bert-base-arabic-camelbert-mix")  # change the model path here to change transfer learning model
    parser = argparse.ArgumentParser(
        description='''evaluates multiple models and assembles data to one results file ''')
    parser.add_argument('--n_fold', required=True, help='N-fold')
    args = parser.parse_args()
    if len(models) == 0:
        models = [
            # local_model_path
            # "aubmindlab/araelectra-base-discriminator",
            # "aubmindlab/araelectra-base-generator",
            "CAMeL-Lab/bert-base-arabic-camelbert-mix",
            # "CAMeL-Lab/bert-base-arabic-camelbert-ca",
            # "bert-base-multilingual-uncased",
            # "bert-base-multilingual-cased",
            # "aubmindlab/bert-base-arabertv2"
        ]
    model_types_dict = {
        "Damith/AraELECTRA-discriminator-SOQAL": "electra",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix": "bert",
        "CAMeL-Lab/bert-base-arabic-camelbert-ca": "bert",
        "aubmindlab/bert-base-arabertv2": "bert",
        "aubmindlab/araelectra-base-discriminator": "electra",
        "aubmindlab/araelectra-base-generator": "electra",
        "bert-base-multilingual-cased": "bert",
        "bert-base-multilingual-uncased": "bert",
        # local models
        ".\\models\\transferlearn\\CAMeL-Lab-bert-base-arabic-camelbert-mix": "bert",
        ".\\models\\transferlearn\\aubmindlab-araelectra-base-discriminator": "electra",
        ".\\models\\transferlearn\\aubmindlab-bert-base-arabertv2": "bert",

    }
    model_predictions = []
    for model in models:
        if model.startswith(".\\models\\transferlearn"):#for local models
            model_name = model.split(os.path.sep)[2]
        else:
            model_name = model.replace('/', '-');
        n_fold_predictions = []
        for i in range(1, int(args.n_fold) + 1):
            seed = i * 777
            output_file = compute(model=model, run_number=i, manual_seed=seed,
                                  model_type=model_types_dict[model],
                                  num_train_epochs=6,  # change these, we got better results at 6 epochs
                                  output_path=model_name + ".json")  # change the output path for transfer learning models.
            n_fold_predictions.append(output_file)
        # model_predictions_output = assemble_results(n_fold_predictions, model.replace('/', '-'))
        model_predictions_output = assemble_results(n_fold_predictions,
                                                    model_name)  # change the output name
        model_predictions.append(model_predictions_output)
    final_results = assemble_results(files=model_predictions, ans_limit=5)
    check_and_evaluate(run_file=final_results, gold_answers_file=raw_dev_set_path)


def compute_permutations():
    local_model_path_camel_mix = os.path.join(".", "models", "transferlearn",
                                              "CAMeL-Lab-bert-base-arabic-camelbert-mix")  # change the model path here to change transfer
    local_model_path_electra_discriminator = os.path.join(".", "models", "transferlearn",
                                                          "aubmindlab-araelectra-base-discriminator")  # change the model path here to change transfer
    local_model_path_arabertv2 = os.path.join(".", "models", "transferlearn",
                                              "aubmindlab-bert-base-arabertv2")  # change the model path here to change transfer
    models = [
        # local_model_path_camel_mix,
        # local_model_path_electra_discriminator,
        # local_model_path_arabertv2
        "Damith/AraELECTRA-discriminator-SOQAL",
        # "aubmindlab/araelectra-base-discriminator",
        # "aubmindlab/araelectra-base-generator",
        # "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        # "CAMeL-Lab/bert-base-arabic-camelbert-ca",
        # "bert-base-multilingual-uncased",
        # "bert-base-multilingual-cased",
        # "aubmindlab/bert-base-arabertv2"d
    ]

    for model in models:
        print("execution : " + model)
        run([model])


if __name__ == '__main__':
    # run()
    compute_permutations()
