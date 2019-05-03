# Knowledge_Graph
Building knowledge graph from input data

# Setup

The following installation steps are written w.r.t. linux operating system and python3 language.

1. Create a new python3 virtual environment:
    `python3 -m venv <path_to_env/env_name>`
2. Switch to the environment:
    `activate path_to_env/env_name/bin/activate`
3. Install Spacy:
    `pip3 install spacy`
4. Install en_core_web_sm model for spacy:
    `python3 -m spacy download en_core_web_sm`
5. Install nltk:
    `pip3 install nltk`
6. Install required nltk data. Either install required packages individually or install all packages by using
    `python -m nltk.downloader all`
    Refer: https://www.nltk.org/data.html
7. Install stanfordcorenlp python package:
    `pip3 install stanfordcorenlp`
8. Download and unzip stanford-corenlp-full: https://stanfordnlp.github.io/CoreNLP/download.html

# Running knowledge_graph.py

Please note that coreference resolution requires around 4GB of free system RAM to run. If this is not available, stanford server may stop with an error or thrashing may cause program to run very slowly.

`python3 knowledge_graphl.py <nltk/stanford/spacy> [ <nltk/stanford/spacy> <nltk/stanford/spacy>]`
 
 nltk runs Named Entity Recognition using custom code written with help of NLTK
 stanford runs NER using stanford's library
 spacy uses spacy's pre-trained models for NER
