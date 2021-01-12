# KnowledgeGraph_Builder

**KnowledgeGraph_Builder** is an end-to-end pipeline for constructing knowledge graphs from unstructured text in the form of RDF triples. It utilizes technologies such as spaCy, Stanford CoreNLP, Neural Coref from Huggingface, and sentence-transformers to perform Named Entity Recognition, Coreference Resolution, Relation Extraction, and Entity-Linking. The goal here is to offer a novel knowledge representation method for applications in automated ontology construction and taxonomy expansion. The intuition here is that by extracting knowledge from text sources, this project will aid in solving the low coverage issue often faced with hand crafted knowledge bases.


## High-Level Pipeline
![pipeline](diagram.png)
- Need to update^^

## Example
**Input Text**:\
Darth Vader, also known by his birth name Anakin Skywalker, is a fictional character in the Star Wars franchise. Darth Vader appears in the original film trilogy as a pivotal antagonist whose actions drive the plot, while his past as Anakin Skywalker and the story of his corruption are central to the narrative of the prequel trilogy. The character was created by George Lucas and has been portrayed by numerous actors. His appearances span the first six Star Wars films, as well as Rogue One, and his character is heavily referenced in Star Wars: The Force Awakens. He is also an important character in the Star Wars expanded universe of television series, video games, novels, literature and comic books. Originally a Jedi who was prophesied to bring balance to the Force, he falls to the dark side of the Force and serves the evil Galactic Empire at the right hand of his Sith master, Emperor Palpatine (also known as Darth Sidious).

**Output Graph**:\
![graph](examples/graph_example.gif)

## Requirements

- `python >=3.7`
- `spacy`
- `pandas`
- `stanford-corenlp`
- `json`
- `nltk`
- [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)


## Setup
1. Create new conda environment:\
   `conda env create -f environment.yml`
2. Activate environment:\
   `conda activate entitylink`
3. Download CoreNLP 4.2.0 and place in root folder of this repo:\
   [`https://stanfordnlp.github.io/CoreNLP/download.html`](https://stanfordnlp.github.io/CoreNLP/download.html)
   

## Usage
1. Add text of your choice in input.txt
2. Run `python text_to_graph.py input.txt`
3. Output will be `output_processed.csv`, containing triples of (entity1, relation, entity2)\

An example of an input and output is included in the `examples` folder


## Example Notebooks
Example notebooks can be found under the `notebooks` folder which contain
- NER_Evaluation.ipynb
- CoRef_Evaluation.ipynb
- Relation_Extraction.ipynb
- EntityLinking_Evaluation.ipynb
- Notebook_Text_to_Graph_Pipeline.ipynb
- Notebook_SpaCy_Parsing_OpenIE_BERT_Evaluation.ipynb


For more information on this project, check out the [Wiki](https://github.com/RhythmSyed/KnowledgeGraph_Builder/wiki/Supporting-Information).
