import os
import sys
from os.path import exists

from examples.arabic.quran.data.preprocess.read_write_qrcd import format_transfer_learning_training_set
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel

raw_training_set = os.path.join(".", "data", "transferlearn", "Arabic-SQuAD.json")
formatted_training_set = os.path.join(".", "data", "transferlearn", "Arabic-SQuAD-Formatted.json")
models_save_path = os.path.join(".", "models", "transferlearn")


def run(
        model_type=None,
        model_name=None,
        manual_seed=777,
        learning_rate=4e-5,
        num_train_epochs=3,
):
    if not exists(formatted_training_set):
        format_transfer_learning_training_set(raw_training_set, formatted_training_set)

    model = QuestionAnsweringModel(
        model_type=model_type,
        model_name=model_name,
        args={"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "learning_rate": learning_rate,
              "manual_seed": manual_seed,
              "num_train_epochs": num_train_epochs},
    )

    model.train_model(formatted_training_set)
    model_save_path = os.path.join(models_save_path, model_name.replace('/', '-'))
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    model.save_model(model_save_path)


if __name__ == '__main__':
    run(model_type='bert', model_name='CAMeL-Lab/bert-base-arabic-camelbert-mix')
