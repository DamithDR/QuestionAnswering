import os
import pandas as pd
from pandas import DataFrame

from simplests.algo.labse import LaBSESTSMethod
from simplests.algo.laser import LASERSTSMethod
from simplests.algo.sbert import SentenceTransformerSTSMethod
from simplests.algo.sif import WordEmbeddingSIFSTSMethod
from simplests.algo.use import UniversalSentenceEncoderSTSMethod
from simplests.model_args import WordEmbeddingSTSArgs, SentenceEmbeddingSTSArgs
import tensorflow_text

training_set_path = os.path.join("examples", "arabic", "quran", "data", "JSON_qrcd_v1.1_train.jsonl")
arcd_path = os.path.join("examples", "arabic", "quran", "data", "flattered-data", "flatteredarcd.json")
squad_path = os.path.join("examples", "arabic", "quran", "data", "flattered-data", "flatteredsquad.json")
arabic_squad_path = os.path.join("examples", "arabic", "quran", "data", "flattered-data", "flatteredarabic_squad.json")
similar_data_dir = os.path.join("examples", "arabic", "quran", "data", "similarity-data")

training = pd.read_json(training_set_path)
arcd = pd.read_json(arcd_path)
squad = pd.read_json(squad_path)
arabic_squad = pd.read_json(arabic_squad_path)

sentence_model_args = SentenceEmbeddingSTSArgs()
sentence_model_args.embedding_model = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
sentence_model_args.language = "en"
universal_model = UniversalSentenceEncoderSTSMethod(model_args=sentence_model_args)

# labse_model_args = SentenceEmbeddingSTSArgs()
# labse_model_args.embedding_model = "https://tfhub.dev/google/LaBSE/2"
# labse_model_args.language = "en"
# labse_model = LaBSESTSMethod(model_args=labse_model_args)
#
# laser_model_args = SentenceEmbeddingSTSArgs()
# laser_model_args.language = "en"
# laser_model = LASERSTSMethod(model_args=laser_model_args)
#
# sbert_model_args = SentenceEmbeddingSTSArgs()
# sbert_model_args.embedding_model = "distiluse-base-multilingual-cased"
# sbert_model_args.language = "en"
# sent_transformer_model = SentenceTransformerSTSMethod(model_args=sbert_model_args)

unique_questions_list = list(set(training['question']))
unique_passages_list = list(set(training['passage']))

# models = [universal_model, labse_model, laser_model, sent_transformer_model]
models = [universal_model]
# model_names = ["universal_model", "labse_model", "laser_model", "sent_transformer_model"]
model_names = ["universal_model"]
# datasets = [arcd, squad, arabic_squad]
datasets = [arabic_squad]

for dset in datasets:
    counter = 0
    for model, model_name in zip(models, model_names):
        predictions = model.fast_predict(ref_set=unique_questions_list, pred_set=dset['question'].to_list())
        for prediction in predictions:
            copy = dset.copy(deep=True)
            copy['score'] = prediction
            copy = copy.sort_values(by=['score'], ascending=False)
            save_path = os.path.join(similar_data_dir, model_name, "ques" + str(counter) + ".tsv")
            limited_df = copy[:100]
            limited_df.to_csv(save_path, sep='\t', index=False)
            counter += 1

# for dset in datasets:
#     counter = 0
#     for qes in unique_questions_list:
#         counter += 1
#         to_predict = []
#         for row in dset['question']:
#             to_predict.append([qes, row])
#         for model, model_name in zip(models, model_names):
#             prediction = model.predict(to_predict)
#             copy = dset.copy(deep=True)
#             copy['score'] = prediction
#             copy = copy.sort_values(by=['score'], ascending=False)
#             save_path = os.path.join(similar_data_dir, model_name, "ques" + str(counter) + ".tsv")
#             limited_df = copy[:100]
#             limited_df.to_csv(save_path, sep='\t', index=False)
#
# for dset in datasets:
#     counter = 0
#     for pas in unique_passages_list:
#         counter += 1
#         to_predict = []
#         for row in dset['passage']:
#             to_predict.append([pas, row])
#         for model, model_name in zip(models, model_names):
#             prediction = model.predict(to_predict)
#             copy = dset.copy(deep=True)
#             copy['score'] = prediction
#             copy = copy.sort_values(by=['score'], ascending=False)
#             save_path = os.path.join(similar_data_dir, model_name, "pass" + str(counter) + ".tsv")
#             limited_df = copy[:100]
#             limited_df.to_csv(save_path, sep='\t', index=False)
