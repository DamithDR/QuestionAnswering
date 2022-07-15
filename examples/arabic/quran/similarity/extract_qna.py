import os.path

import pandas as pd

from examples.arabic.quran.data.preprocess.read_write_qrcd import PassageQuestion, write_to_JSONL_file

arcd = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "arcd.json")
squad = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "squad-train-v2.0.json")
arabic_squad = os.path.join("examples", "arabic", "quran", "data", "transferlearn", "Arabic-SQuAD.json")
file_list = [arcd, squad, arabic_squad]
file_name_list = ["arcd", "squad", "arabic_squad"]

passage_question_lists = {}
for file_path, file_name in zip(file_list, file_name_list):
    unique_no = 1
    pq_list = []
    dataframe = pd.read_json(file_path)
    data_list = dataframe['data'].tolist()
    print("processing " + file_name)
    for data in data_list:
        paragraphs = data['paragraphs']
        for para in paragraphs:
            passage = para['context']
            for qa in para["qas"]:
                question = qa['question']
                ans_list = []
                for ans in qa['answers']:
                    ans_dict = {"text": ans['text'], "start_char": ans['answer_start']}
                    ans_list.append(ans_dict)

                pq_dict = {"pq_id": (file_name + str(unique_no)), "passage": passage, "question": question,
                           "answers": ans_list, "surah": 1, "verses": "1"}
                pq = PassageQuestion(pq_dict)
                pq_list.append(pq)
                unique_no += 1
        passage_question_lists[file_name] = pq_list

for file_name in file_name_list:
    write_to_JSONL_file(passage_question_lists[file_name],
                        "examples/arabic/quran/data/flattered-data/" + "flattered" + file_name + ".json")

print("done")
