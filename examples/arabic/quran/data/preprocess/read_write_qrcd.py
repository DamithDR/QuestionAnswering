"""
A script to:
 - read a JSONL (JSON Lines) dataset into objects of class PassageQuestion
 - write the PassageQuestion objects to another JSONL file

"""
import json, argparse




def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


class Qas():
    ansId = 0

    def __init__(self, dictionary, pq_id) -> None:
        self.id = None
        self.is_impossible = None
        self.question = None
        self.answers = []
        Qas.ansId += 1
        if pq_id is None:
            self.id = '{:0>5}'.format(self.ansId)
        else:
            self.id = pq_id
        self.question = dictionary["question"]
        for answer in dictionary["answers"]:
            self.answers.append(Answer(answer))
        if len(self.answers) > 0:
            self.is_impossible = False
        else:
            self.is_impossible = True

    def to_dict(self, diacritized=False) -> dict:
        if diacritized:
            qa_dict = {
                "id": self.id,
                "is_impossible": self.is_impossible,
                # "question": farasa_diacritizer.diacritize(self.question), commented out as farasapy breaks the execution, uncomment if you want to diacritize the data
                "answers": [ans.to_dict_formatted(diacritized) for ans in self.answers]
            }
        else:
            qa_dict = {
                "id": self.id,
                "is_impossible": self.is_impossible,
                "question": self.question,
                "answers": [ans.to_dict_formatted() for ans in self.answers]
            }
        return qa_dict

    def to_test_dict(self, diacritized=False) -> dict:
        if diacritized:
            qa_dict = {
                "id": self.id,
                # "question": farasa_diacritizer.diacritize(self.question),
            }
        else:
            qa_dict = {
                "id": self.id,
                "question": self.question,
            }
        return qa_dict


class TransferLearnQas():
    ansId = 0

    def __init__(self, dictionary) -> None:
        self.id = None
        self.is_impossible = None
        self.question = None
        self.answers = []
        # Qas.ansId += 1
        # if pq_id is None:
        #     self.id = '{:0>5}'.format(self.ansId)
        # else:
        #     self.id = pq_id
        self.id = dictionary['id']
        self.question = dictionary["question"]
        for answer in dictionary["answers"]:
            self.answers.append(TransferLearnAnswer(answer))
        if len(self.answers) > 0:
            self.is_impossible = False
        else:
            self.is_impossible = True

    def to_dict(self) -> dict:
        qa_dict = {
            "id": self.id,
            "is_impossible": self.is_impossible,
            "question": self.question,
            "answers": [ans.to_dict_formatted() for ans in self.answers]
        }
        return qa_dict


class Answer():
    def __init__(self, dictionary) -> None:
        self.text = dictionary["text"]
        self.start_char = dictionary["start_char"]

    def to_dict(self) -> dict:
        answer_dict = {
            "text": self.text,
            "start_char": self.start_char
        }
        return answer_dict

    def to_dict_formatted(self, diacritized=False) -> dict:
        if diacritized:
            answer_dict = {
                # "text": farasa_diacritizer.diacritize(self.text),
                "answer_start": self.start_char
            }
        else:
            answer_dict = {
                "text": self.text,
                "answer_start": self.start_char
            }
        return answer_dict


class TransferLearnAnswer():
    def __init__(self, dictionary) -> None:
        self.text = dictionary["text"]
        self.answer_start = dictionary["answer_start"]

    def to_dict(self) -> dict:
        answer_dict = {
            "text": self.text,
            "answer_start": self.answer_start
        }
        return answer_dict

    def to_dict_formatted(self) -> dict:
        answer_dict = {
            "text": self.text,
            "answer_start": self.answer_start
        }
        return answer_dict


class PassageQuestion():
    def __init__(self, dictionary) -> None:
        self.pq_id = None
        self.passage = None
        self.surah = None
        self.verses = None
        self.question = None
        self.answers = []
        self.pq_id = dictionary["pq_id"]
        self.passage = dictionary["passage"]
        self.surah = dictionary["surah"]
        self.verses = dictionary["verses"]
        self.question = dictionary["question"]
        for answer in dictionary["answers"]:
            self.answers.append(Answer(answer))

    def to_dict(self) -> dict:
        passge_question_dict = {
            "pq_id": self.pq_id,
            "passage": self.passage,
            "surah": self.surah,
            "verses": self.verses,
            "question": self.question,
            "answers": [x.to_dict() for x in self.answers]
        }
        return passge_question_dict


class FormattedPassageQuestion():
    def __init__(self, dictionary) -> None:
        self.passage = None
        self.pq_id = None
        self.qas = []
        self.passage = dictionary["passage"]
        self.pq_id = dictionary["pq_id"]
        self.qas.append(Qas(dictionary, self.pq_id), )

    def to_dict(self, diacritized) -> dict:
        if diacritized:
            passge_question_dict = {
                "pq_id": self.pq_id,
                # "context": farasa_diacritizer.diacritize(self.passage),
                "qas": [x.to_dict(diacritized) for x in self.qas]
            }
        else:
            passge_question_dict = {
                "pq_id": self.pq_id,
                "context": self.passage,
                "qas": [x.to_dict() for x in self.qas]
            }
        return passge_question_dict

    def to_test_dict(self, diacritized=False) -> dict:
        if diacritized:
            passge_question_dict = {
                "pq_id": self.pq_id,
                # "context": farasa_diacritizer.diacritize(self.passage),
                "qas": [x.to_test_dict(diacritized) for x in self.qas]
            }
        else:
            passge_question_dict = {
                "pq_id": self.pq_id,
                "context": self.passage,
                "qas": [x.to_test_dict() for x in self.qas]
            }
        return passge_question_dict


class FormattedTransferLearnQuestion():
    def __init__(self, dictionary) -> None:
        self.passage = None
        self.pq_id = None
        self.qas = []
        self.passage = dictionary["context"]
        # self.pq_id = dictionary["id"]
        for qa in dictionary['qas']:
            self.qas.append(TransferLearnQas(qa))

    def to_dict(self) -> dict:
        passge_question_dict = {
            # "pq_id": self.pq_id,
            "context": self.passage,
            "qas": [x.to_dict() for x in self.qas]
        }
        return passge_question_dict

    def to_test_dict(self) -> dict:
        passge_question_dict = {
            "pq_id": self.pq_id,
            "context": self.passage,
            "qas": [x.to_test_dict() for x in self.qas]
        }
        return passge_question_dict


def read_JSONL_file(file_path) -> list:
    data_in_file = load_jsonl(file_path)

    # get list of PassageQuestion objects
    passage_question_objects = []
    for passage_question_dict in data_in_file:
        # instantiate a PassageQuestion object
        pq_object = PassageQuestion(passage_question_dict)
        passage_question_objects.append(pq_object)

    print(f"Collected {len(passage_question_objects)} Object from {file_path}")
    return passage_question_objects


def read_JSONL_file_formatted(file_path) -> list:
    data_in_file = load_jsonl(file_path)

    # get list of PassageQuestion objects
    passage_question_objects = []
    for passage_question_dict in data_in_file:
        # instantiate a PassageQuestion object
        pq_object = FormattedPassageQuestion(passage_question_dict)
        passage_question_objects.append(pq_object)

    print(f"Collected {len(passage_question_objects)} Object from {file_path}")
    return passage_question_objects


def read_JSONL_file_transfer_learning_formatted(file_path) -> list:
    data_in_file = load_jsonl(file_path)

    # get list of PassageQuestion objects
    passage_question_objects = []
    data = data_in_file[0]['data']
    for obj in data:
        paragraphs = obj['paragraphs']
        for para in paragraphs:
            pq_object = FormattedTransferLearnQuestion(para)
            passage_question_objects.append(pq_object)

    print(f"Collected {len(passage_question_objects)} Object from {file_path}")
    return passage_question_objects


def write_to_JSONL_file(passage_question_objects, output_path, diacritized=False) -> None:
    # list of dictionaries for the passage_question_objects
    dict_data_list = []
    for pq_object in passage_question_objects:
        dict_data = pq_object.to_dict()
        dict_data_list.append(dict_data)
    dump_jsonl(dict_data_list, output_path)


def write_test_set_to_JSONL_file(passage_question_objects, output_path, diacritized=False) -> None:
    # list of dictionaries for the passage_question_objects
    dict_data_list = []
    for pq_object in passage_question_objects:
        dict_data = pq_object.to_test_dict(diacritized)
        dict_data_list.append(dict_data)
    dump_jsonl(dict_data_list, output_path)


def format_training_set(input_file, output_file):
    pq_objects = read_JSONL_file_formatted(input_file)
    write_to_JSONL_file(pq_objects, output_file, diacritized=False)


def format_transfer_learning_training_set(input_file, output_file):
    pq_objects = read_JSONL_file_transfer_learning_formatted(input_file)
    write_to_JSONL_file(pq_objects, output_file, diacritized=False)


def format_n_diacritize_training_set(input_file, output_file):
    pq_objects = read_JSONL_file_formatted(input_file)
    write_to_JSONL_file(pq_objects, output_file, diacritized=True)


def format_dev_set(input_file, output_file):
    pq_objects = read_JSONL_file_formatted(input_file)
    write_test_set_to_JSONL_file(pq_objects, output_file, diacritized=False)


def format_n_diacritize_dev_set(input_file, output_file):
    pq_objects = read_JSONL_file_formatted(input_file)
    write_test_set_to_JSONL_file(pq_objects, output_file, diacritized=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Copying data from input file and storing it as list of object and writing it in file''')
    parser.add_argument('--input_file', required=True, help='Input File')
    parser.add_argument('--output_file', required=False, help='Output File', default="collected_file.jsonl")
    args = parser.parse_args()
    args_as_dict = vars(args)
    # passage_question_objects = read_JSONL_file(args_as_dict['input_file'])
    passage_question_objects = read_JSONL_file_formatted(args_as_dict['input_file'])
    write_to_JSONL_file(passage_question_objects, args_as_dict['output_file'])
