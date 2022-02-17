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

    def __init__(self, dictionary) -> None:
        self.id = None
        self.is_impossible = None
        self.question = None
        self.answers = []
        Qas.ansId += 1
        self.id = '{:0>5}'.format(self.ansId)
        self.question = dictionary["question"]
        for answer in dictionary["answers"]:
            self.answers.append(Answer(answer))
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

    def to_dict_formatted(self) -> dict:
        answer_dict = {
            "text": self.text,
            "answer_start": self.start_char
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
        self.qas = []
        self.passage = dictionary["passage"]
        self.qas.append(Qas(dictionary))

    def to_dict(self) -> dict:
        passge_question_dict = {
            "context": self.passage,
            "qas": [x.to_dict() for x in self.qas]
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


def write_to_JSONL_file(passage_question_objects, output_path) -> None:
    # list of dictionaries for the passage_question_objects
    dict_data_list = []
    for pq_object in passage_question_objects:
        dict_data = pq_object.to_dict()
        dict_data_list.append(dict_data)
    dump_jsonl(dict_data_list, output_path)


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
