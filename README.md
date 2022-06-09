
# QuestionAnswering : Qur'an Question Answering with Transformers

Transformers based approach for question answering in Qur'an which employs transfer-learning, ensemble-learning across multiple models.

## Installation
You first need to install Java for the evaluation script which uses `farasapy` and the desired version is Java8. 
Please refer [Oracle installation guide](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html) for more details on installing JDK for different platforms.

Then you need to install PyTorch. The recommended PyTorch version is 1.11.0
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for more details specifically for the platforms.

When PyTorch has been installed, you can install requirements from source by cloning the repository and running:

```bash
git clone https://github.com/DamithDR/QuestionAnswering.git
cd QuestionAnswering
pip install -r requirements.txt
```

## Experiment Results
You can easily run experiments using following command and altering the parameters as you wish

```bash
python -m examples.arabic.quran.quran_question_answering --n_fold=1 --transfer_learning=False --self_ensemble=False --models=camelmix,arabert
```

## Experiment using Docker
To run using docker, you need to have docker installed in your machine. Please use [Docker installation Guide](https://docs.docker.com/get-docker/) to install docker based on your operating system.

Once you successfully installed docker in your system, you can simply use following command to execute the experiments.
```bash
docker run damithpremasiri/question-answering-quran:v-1.0 --n_fold=1 --transfer_learning=False --self_ensemble=False --models=camelmix,arabert
```

## Parameters
Please find the detailed descriptions of the parameters
```text
n_fold              : Number of executions expected before self ensemble
transfer_learning   : On/Off transfer learning
self_ensemble       : On/Off self ensembling
models              : comma seperated model tags
```

## Model Tags
```text
arabert             : aubmindlab/bert-base-arabertv2
mbertcased          : bert-base-multilingual-cased
mbertuncased        : bert-base-multilingual-uncased
camelmix            : CAMeL-Lab/bert-base-arabic-camelbert-mix
camelca             : CAMeL-Lab/bert-base-arabic-camelbert-ca
araelectradisc      : aubmindlab/araelectra-base-discriminator
araelectragen       : aubmindlab/araelectra-base-generator
```

## Citation
Please consider citing us if you use the library or the code. 
```bash
@inproceedings{damith2022DTWquranqa,
  title={DTW at Qur'an QA 2022: Utilising Transfer Learning with Transformers for Question Answering in a Low-resource Domain},
  author={Damith Premasiri and Tharindu Ranasinghe and Wajdi Zaghouani and Ruslan Mitkov},
  booktitle={Proceedings of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT5).},
  year={2022}
}
```
