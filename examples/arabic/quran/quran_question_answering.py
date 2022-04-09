import argparse
import json
import os
import sys
from os.path import exists

import torch
from pyarabic import araby

from examples.arabic.quran import assembler
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_training_set, format_n_diacritize_training_set, \
    format_n_diacritize_dev_set
from examples.arabic.quran.data.preprocess.read_write_qrcd import load_jsonl
from examples.arabic.quran import quranqa22_eval
from examples.arabic.quran import transfer_learning_model_generation
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel

model_types_dict = {
    "Damith/AraELECTRA-discriminator-QuranQA": "electra",
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

model_tags = {
    "arabert": "aubmindlab/bert-base-arabertv2",
    "mbertcased": "bert-base-multilingual-cased",
    "mbertuncased": "bert-base-multilingual-uncased",
    "camelmix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "camelca": "CAMeL-Lab/bert-base-arabic-camelbert-ca",
    "araelectradisc": "aubmindlab/araelectra-base-discriminator",
    "araelectragen": "aubmindlab/araelectra-base-generator",
    "araelectraquran": "Damith/AraELECTRA-discriminator-QuranQA"
}


def validate_and_parse_models(models=None):
    if models is None:
        return []

    models_list = str(models).split(',')
    models = []
    for model in models_list:
        if model in model_tags:
            models.append(model_tags[model])
    return models


def compute(
        learning_rate=4e-5,
        num_train_epochs=6,
        manual_seed=None,
        model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix",
        model_type="bert",
        diacritize=False,
        run_number=1,
        output_path=None
):
    raw_training_set_path = os.path.join("examples", "arabic", "quran", "data", "qrcd_v1.1_train.jsonl")
    training_set_path = os.path.join("examples", "arabic", "quran", "data", "preprocess", "output", "qrcd_v1.1_train_formatted.jsonl")
    raw_dev_set_path = os.path.join("examples", "arabic", "quran", "data", "qrcd_v1.1_dev.jsonl")
    # raw_dev_set_path = os.path.join("examples", "arabic", "quran", "data", "qrcd_v1.1_test_noAnswers.jsonl")
    dev_set_path = os.path.join("examples", "arabic", "quran", "data", "preprocess", "output", "qrcd_v1.1_dev_formatted.jsonl")
    # dev_set_path = os.path.join("examples", "arabic", "quran", "data", "preprocess", "output", "qrcd_v1.1_test_noAnswers_formatted.jsonl")
    results_folder = os.path.join("examples", "arabic", "quran", "data", "run-files")
    file_no = len(
        [name for name in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder, name))]) + 1
    if output_path is None:
        results_file = os.path.join("examples", "arabic", "quran", "data", "run-files", "assemble",
                                    model_name.replace('/', '-') + str(run_number) + ".json")
    else:
        results_file = os.path.join("examples", "arabic", "quran", "data", "run-files", "assemble", output_path)

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

    qa_model = QuestionAnsweringModel(
        model_type=model_type,
        model_name=model_name,
        args={"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "learning_rate": learning_rate,
              "manual_seed": manual_seed,
              "num_train_epochs": num_train_epochs,
              "save_eval_checkpoints": False,
              "save_model_every_epoch": False
              },
        use_cuda=torch.cuda.is_available()
    )

    # Train the model
    qa_model.train_model(training_set_path)
    # make predictions
    dev_set = load_jsonl(dev_set_path)

    answers, scores, ans_with_scores = qa_model.predict(dev_set, n_best_size=5)
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


if __name__ == '__main__':
    PYTHON_PATH = "D:\\SharedTasks\\QA-Arab\\QuestionAnswering"
    sys.path.append(PYTHON_PATH)

    gold_answers_path = os.path.join("examples", "arabic", "quran", "data", "qrcd_v1.1_dev.jsonl")
    transfer_learning_model_path = os.path.join(".", "outputs")

    parser = argparse.ArgumentParser(
        description='''evaluates multiple models and assembles data to one result which can be output ''')
    parser.add_argument('--n_fold', required=False, help='N-fold', default=1)
    parser.add_argument('--transfer_learning', required=False, help='Enable Transfer Learning', default=False)
    parser.add_argument('--self_ensemble', required=False, help='Enable Self Ensembling', default=False)
    parser.add_argument('--models', required=False, help='Models to be used')
    args = parser.parse_args()
    n_fold = int(args.n_fold)
    if args.transfer_learning == "True":
        transfer_learning = True
    else:
        transfer_learning = False
    if args.self_ensemble == "True":
        self_ensemble = args.self_ensemble
    else:
        self_ensemble = False
    parsed_models = validate_and_parse_models(args.models)

    print('n_fold = {0} | transfer_learning = {1} | self_ensemble = {2} | parsed_models = {3}'
          .format(n_fold, transfer_learning, self_ensemble, parsed_models))

    if self_ensemble is False & n_fold > 1:
        print("n_fold defaulting to 1 as self_ensemble is false. "
              "Please use --self_ensemble=True if you want to ensemble results")
        n_fold = 1
    if len(parsed_models) == 0:
        print("No models provided, transfer_learning defaulting to False, "
              "using default model Damith/AraELECTRA-discriminator-QuranQA \n"
              "Please use --models=modeltag1,modeltag2 to use models.\n"
              "Model Tags : Model\n"
              "arabert : aubmindlab/bert-base-arabertv2\n"
              "mbertcased: bert-base-multilingual-cased\n"
              "mbertuncased: bert-base-multilingual-uncased\n"
              "camelmix: CAMeL-Lab/bert-base-arabic-camelbert-mix\n"
              "camelca: CAMeL-Lab/bert-base-arabic-camelbert-ca\n"
              "araelectradisc: aubmindlab/araelectra-base-discriminator\n"
              "araelectragen: aubmindlab/araelectra-base-generator"
              )
        parsed_models.append("Damith/AraELECTRA-discriminator-SOQAL")
        transfer_learning = False
    models = parsed_models

    model_predictions = []
    for mdl in models:
        if mdl.startswith(".\\models\\transferlearn"):  # for local models
            model_reference = mdl.split(os.path.sep)[2]
        else:
            model_reference = mdl.replace('/', '-')
        if transfer_learning & (mdl != model_tags['araelectraquran']):  # this one is already transferlearned
            transfer_learning_model_generation.generate_model(model_type=model_types_dict[mdl], model_name=mdl)
        n_fold_predictions = []
        for i in range(1, n_fold + 1):
            seed = i * 777
            if transfer_learning:
                output_file = compute(model_name=transfer_learning_model_path, run_number=i, manual_seed=seed,
                                      model_type=model_types_dict[mdl],
                                      num_train_epochs=6,  # change these, we got better results at 6 epochs
                                      output_path=model_reference + ".json")
            else:
                output_file = compute(model_name=mdl, run_number=i, manual_seed=seed,
                                      model_type=model_types_dict[mdl],
                                      num_train_epochs=6,  # change these, we got better results at 6 epochs
                                      output_path=model_reference + ".json")
            n_fold_predictions.append(output_file)
        # model_predictions_output = assembler.assemble_results(n_fold_predictions, model.replace('/', '-'))
        model_predictions_output = assembler.assemble_results(n_fold_predictions,
                                                              model_reference)  # change the output name
        model_predictions.append(model_predictions_output)
    final_results = assembler.assemble_results(files=model_predictions, ans_limit=5)
    quranqa22_eval.check_and_evaluate(run_file=final_results, gold_answers_file=gold_answers_path)
