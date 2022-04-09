import os
import sys
from os.path import exists

import torch

from examples.arabic.quran.data.preprocess import read_write_qrcd
from questionanswering.transformers.question_answering_model import QuestionAnsweringModel

raw_training_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "Arabic-SQuAD.json")
formatted_training_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn",
                                      "Arabic-SQuAD-Formatted.json")
# formatted_training_set = os.path.join("examples", "arabic", "quran", "data"
# , "transferlearn","temp", "Arabic-SQuAD-Formatted.json")
models_save_path = os.path.join("examples", "arabic", "quran", "models", "transferlearn")


def generate_model(
        model_type=None,
        model_name=None,
        manual_seed=777,
        learning_rate=4e-5,
        num_train_epochs=5, # change
        train_batch_size=8,  # change this according to your GPU
):
    if not exists(formatted_training_set):
        read_write_qrcd.format_transfer_learning_training_set(raw_training_set, formatted_training_set)

    model = QuestionAnsweringModel(
        model_type=model_type,
        model_name=model_name,
        args={"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "learning_rate": learning_rate,
              "manual_seed": manual_seed,
              "train_batch_size": train_batch_size,
              "num_train_epochs": num_train_epochs,
              "save_eval_checkpoints": False,
              "save_model_every_epoch": False},
        use_cuda=torch.cuda.is_available()
    )

    model.train_model(formatted_training_set)
    model.save_model()


if __name__ == '__main__':
    # run(model_type='bert', model_name='CAMeL-Lab/bert-base-arabic-camelbert-mix')
    # run(model_type='electra', model_name='aubmindlab/araelectra-base-discriminator')
    generate_model(model_type='bert', model_name='aubmindlab/bert-base-arabertv2')
