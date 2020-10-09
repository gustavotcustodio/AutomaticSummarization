# Automatic Summarization of Scientific Papers for Generating Research Highlights

This project provides a new representation of text data in form of sentence embeddings called SaKS (Sections and Keywords Similarities). The attributes in this representation are generated using information from sections of scientific papers.

This repository provides:

* Data from 199 scientific papers in text form, word embeddings and in SaKS format.
* Scripts for running experiments with data classifiers, such as K-NN and SVMs.
* Modified versions of the Refresh model to run our dataset.

The Refresh model and its original data is available in: https://github.com/EdinburghNLP/Refresh

The pre-trained word embeddings trained on "1 billion word language modeling benchmark r13output" is available in [this link](http://bollin.inf.ed.ac.uk/public/direct/Refresh-NAACL18-1-billion-benchmark-wordembeddings.tar.gz).
