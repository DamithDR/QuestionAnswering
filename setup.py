from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="quesans",
    version="0.0.1",
    author="Damith Premasiri",
    author_email="damithpremasiri@gmail.com",
    description="Deep Learning based Questions Answering package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DamithDR/QuestionAnswering",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "transformers==4.16.2",
        "tensorboard",
        "datasets==1.18.3",
        "numpy==1.22.2",
        "pandas==1.4.0",
        "tqdm==4.62.3",
        "farasapy",
        "PyArabic~=0.6.14"
    ],
)