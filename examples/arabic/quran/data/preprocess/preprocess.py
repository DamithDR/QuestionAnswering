import os
from os.path import exists

from examples.arabic.quran.data.preprocess import read_write_qrcd
from examples.arabic.quran.data.preprocess.read_write_qrcd import format_dev_set

raw_training_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "Arabic-SQuAD.json")
formatted_training_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn",
                                      "Arabic-SQuAD-Formatted.json")
arcd_file = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "arcd.json")
formatted_arcd_file = os.path.join("examples", "arabic", "quran", "data", "transferlearn",
                                   "arcd-Formatted.json")

raw_test_set_path = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "qrcd_v1.1_test_gold.jsonl")
formatted_test_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn",
                                  "qrcd_v1.1_test_gold_formatted.json")

raw_squad_set_path = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "squad-train-v2.0.json")
formatted_squad_test_set = os.path.join("examples", "arabic", "quran", "data", "transferlearn",
                                  "squad-train-v2.0_formatted.json")

# if not exists(formatted_training_set):
#     read_write_qrcd.format_transfer_learning_training_set(raw_training_set, formatted_training_set)

if not exists(formatted_arcd_file):
    read_write_qrcd.format_transfer_learning_training_set(arcd_file, formatted_arcd_file)

if not exists(formatted_test_set):
    format_dev_set(raw_test_set_path, formatted_test_set)

if not exists(formatted_squad_test_set):
    read_write_qrcd.format_transfer_learning_training_set(raw_squad_set_path, formatted_squad_test_set)
