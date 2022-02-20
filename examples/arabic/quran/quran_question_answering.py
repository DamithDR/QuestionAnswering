import json
import os

from questionanswering.transformers.question_answering_model import QuestionAnsweringModel


# Create dummy data to use for training.
# train_data = [
#     {
#         "context": "This is the first context",
#         "qas": [
#             {
#                 "id": "00001",
#                 "is_impossible": False,
#                 "question": "Which context is this?",
#                 "answers": [{"text": "the first", "answer_start": 8}],
#             }
#         ],
#     },
#     {
#         "context": "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,            and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
#         "qas": [
#             {
#                 "id": "00002",
#                 "is_impossible": False,
#                 "question": "What was the cost to society?",
#                 "answers": [{"text": "low cost", "answer_start": 225}],
#             },
#             {
#                 "id": "00003",
#                 "is_impossible": False,
#                 "question": "What was the name of the 1937 treaty?",
#                 "answers": [{"text": "Bald Eagle Protection Act", "answer_start": 167}],
#             },
#         ],
#     },
# ]
#
# # Save as a JSON file
# os.makedirs("data", exist_ok=True)
# with open("data/train.json", "w") as f:
#     json.dump(train_data, f)

def run(
        learning_rate=4e-5,
        num_train_epochs=1,
        manual_seed=None,
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix"
):
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
    model.train_model(".\data\preprocess\output\qrcd_v1.1_train_formatted.jsonl")

    # Evaluate the model. (Being lazy and evaluating on the train data itself)
    result, text = model.eval_model(".\data\preprocess\output\qrcd_v1.1_dev_formatted.jsonl")

    print(result)
    print(text)

    print("-------------------")

    return result, text


if __name__ == '__main__':
    run()
