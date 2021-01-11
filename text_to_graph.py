import spacy
import pandas as pd
import en_core_web_lg
from stanfordcorenlp import StanfordCoreNLP
import json
from collections import defaultdict
import nltk
import sys
from docopt import docopt
from openie import StanfordOpenIE
from sentence_transformers import SentenceTransformer, util


def process_NER(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    ner_dict = {}
    for x in doc.ents:
        ner_dict[x.text] = x.label_
    return ner_dict


def process_corefs(text):
    nlp = StanfordCoreNLP("./stanford-corenlp-4.2.0", quiet=False)
    annotated = nlp.annotate(text, properties={'annotators': 'coref', 'pipelineLanguage': 'en'})
    result = json.loads(annotated)
    corefs = result['corefs']
    return corefs


def process_dependency_matching(text, ner_dict, corefs):
    replace_coref_with = []
    sentence_wise_replacements = defaultdict(list)
    sentences = nltk.sent_tokenize(text)

    for index, coreferences in enumerate(corefs.values()):
        replace_with = coreferences[0]
        for reference in coreferences:
            if reference["text"] in ner_dict.keys() or reference["text"][reference["headIndex"] - reference["startIndex"]] in ner_dict.keys():
                replace_with = reference
            sentence_wise_replacements[reference["sentNum"] - 1].append((reference, index))
        replace_coref_with.append(replace_with["text"])
    sentence_wise_replacements[0].sort(key=lambda tup: tup[0]["startIndex"])

    # Carry out replacement
    for index, sent in enumerate(sentences):
        replacement_list = sentence_wise_replacements[index]
        for item in replacement_list[::-1]:
            to_replace = item[0]
            replace_with = replace_coref_with[item[1]]
            replaced_sent = ""
            words = nltk.word_tokenize(sent)

            for i in range(len(words) - 1, to_replace["endIndex"] - 2, -1):
                replaced_sent = words[i] + " " + replaced_sent

            replaced_sent = replace_with + " " + replaced_sent

            for i in range(to_replace["startIndex"] - 2, -1, -1):
                replaced_sent = words[i] + " " + replaced_sent

            # print(replaced_sent)
            sentences[index] = replaced_sent
    result = ""
    for sent in sentences:
        result += sent
    return result


def process_relation_extraction(text):
    triples = []
    with StanfordOpenIE() as client:
        for triple in client.annotate(text):
            triples.append(triple)

    return pd.DataFrame(triples)


def process_triple_pruning(triples, ner_dict):
    entity_set = set(ner_dict.keys())
    final_triples = []

    for row, col in triples.iterrows():
        col['subject'] = col['subject'].strip()

        # check if Named Entity in subject sentence fragment
        found_entity = False
        for named_entity in entity_set:
            if named_entity in col['subject']:
                col['subject'] = named_entity
                found_entity = True

        if found_entity:
            final_triples.append(('Node', col['subject'], col['relation'], 'Node', col['object']))

    triple_df = pd.DataFrame(final_triples, columns=['Type1', 'Entity1', 'Relationship', 'Type2', 'Entity2']).drop_duplicates()
    return triple_df


def process_entity_linking(triple_df, confidence_threshold):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    index = 0
    for _, col1 in triple_df.iterrows():
        head = col1['Entity2']
        embedding1 = model.encode(head, convert_to_tensor=True)

        for _, col2 in triple_df.iterrows():
            tail = col2['Entity2']
            if head == tail:
                continue

            embedding2 = model.encode(tail, convert_to_tensor=True)
            confidence = util.pytorch_cos_sim(embedding1, embedding2)[0][0]

            if confidence > confidence_threshold:  # 85% seems to work pretty well
                # Perform logic for linking
                new_tail = head if len(tail) < len(head) else tail

                col1['Entity2'] = new_tail
                col2['Entity2'] = new_tail

                print("Sentence 1:", head)
                print("Sentence 2:", tail)
                print("Similarity:", confidence)
                index += 1
                print('Processed {} out of {}'.format(index, len(triple_df)))
                print()

    triple_df = triple_df.drop_duplicates()
    return triple_df


def main():
    args = docopt("""KnowledgeGraph_Builder
        Usage:
            text_to_graph.py <input_text>

            <input_text> = input text to be processed as RDF triples
        """)

    input_file = args['<input_text>']

    # Read input text file
    with open(input_file, "r") as f:
        lines = f.readlines()

    input_text = ""
    for line in lines:
        input_text += line

    # Perform Named Entity Recognition with spaCy
    ner_dict = process_NER(text=input_text)
    print('***** Completed NER *****')

    # Generate Coreferences and Dependencies with CoreNLP
    corefs = process_corefs(text=input_text)
    print("Coreferences found: ", len(corefs))
    print("Named entities: ", ner_dict.keys())

    # Perform Replacement with Named Entities and Dependencies
    resolved_text = process_dependency_matching(text=input_text, ner_dict=ner_dict, corefs=corefs)
    print('***** Completed Coreference Resolution *****')

    # Perform Relation Extraction using Stanford OpenIE
    triples = process_relation_extraction(text=resolved_text)
    print('***** Completed Relation Extraction *****')

    # Perform pruning with Named Entity and Triple matching
    triple_df = process_triple_pruning(triples=triples, ner_dict=ner_dict)
    print('***** Completed Pruning *****')

    # Perform Entity Linking between nodes
    triple_df = process_entity_linking(triple_df=triple_df, confidence_threshold=0.70)
    print('***** Completed Entity Linking *****')

    # Write to csv
    triple_df.to_csv('output_processed.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
