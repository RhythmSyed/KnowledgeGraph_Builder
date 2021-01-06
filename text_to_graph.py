import spacy
import pandas as pd
import en_core_web_lg
from stanfordcorenlp import StanfordCoreNLP
import json
from collections import defaultdict
import nltk


starwars_text = 'Darth Vader, also known by his birth name Anakin Skywalker, is a fictional character in the Star Wars franchise. Darth Vader appears in the original film trilogy as a pivotal antagonist whose actions drive the plot, while his past as Anakin Skywalker and the story of his corruption are central to the narrative of the prequel trilogy. The character was created by George Lucas and has been portrayed by numerous actors. His appearances span the first six Star Wars films, as well as Rogue One, and his character is heavily referenced in Star Wars: The Force Awakens. He is also an important character in the Star Wars expanded universe of television series, video games, novels, literature and comic books. Originally a Jedi prophesied to bring balance to the Force, he falls to the dark side of the Force and serves the evil Galactic Empire at the right hand of his Sith master, Emperor Palpatine (also known as Darth Sidious).'

nlp = en_core_web_lg.load()
doc = nlp(starwars_text)

ner_dict = {}
for x in doc.ents:
    ner_dict[x.text] = x.label_


nlp = StanfordCoreNLP("./stanford-corenlp-4.2.0", quiet=False)
annotated = nlp.annotate(starwars_text, properties={'annotators': 'coref', 'pipelineLanguage': 'en'})
result = json.loads(annotated)


corefs = result['corefs']
print("Coreferences found: ",len(corefs))
print("Named entities: " , ner_dict.keys())

replace_coref_with = []
sentence_wise_replacements = defaultdict(list)

sentences = nltk.sent_tokenize(starwars_text)

print('Number of sentences: ', len(sentences))
print(sentences)


for index,coreferences in enumerate(corefs.values()):
    replace_with = coreferences[0]
    for reference in coreferences:
        if reference["text"] in ner_dict.keys() or reference["text"][reference["headIndex"]-reference["startIndex"]] in ner_dict.keys():
            replace_with = reference
        sentence_wise_replacements[reference["sentNum"]-1].append((reference,index))
    replace_coref_with.append(replace_with["text"])

sentence_wise_replacements[0].sort(key=lambda tup: tup[0]["startIndex"])

tokenizer = nltk.word_tokenize

# Carry out replacement
for index, sent in enumerate(sentences):
    replacement_list = sentence_wise_replacements[index]  # replacement_list : [({},int)]
    for item in replacement_list[::-1]:  # item : ({},int)
        to_replace = item[0]  # to_replace: {}
        replace_with = replace_coref_with[item[1]]
        replaced_sent = ""
        words = tokenizer(sent)

        for i in range(len(words) - 1, to_replace["endIndex"] - 2, -1):
            replaced_sent = words[i] + " " + replaced_sent

        replaced_sent = replace_with + " " + replaced_sent

        for i in range(to_replace["startIndex"] - 2, -1, -1):
            replaced_sent = words[i] + " " + replaced_sent

        sentences[index] = replaced_sent

result = ""
for sent in sentences:
    result += sent


from openie import StanfordOpenIE

triples = []
with StanfordOpenIE() as client:
    for triple in client.annotate(result):
        triples.append(triple)

triples = pd.DataFrame(triples)


entity_set = set(ner_dict.keys())

final_triples = []
for row, col in triples.iterrows():
    col['subject'] = col['subject'].strip()

    if col['subject'] in entity_set:
        added = False
        entity2_sent = col['object']
        for entity in entity_set:
            if entity in entity2_sent:
                final_triples.append(
                    (ner_dict[col['subject']], col['subject'], col['relation'], ner_dict[entity], col['object']))
                added = True
        if not added:
            final_triples.append((ner_dict[col['subject']], col['subject'], col['relation'], 'O', col['object']))


final_df = pd.DataFrame(final_triples, columns=['Type', 'Head', 'Relationship', 'Type', 'Tail'])
final_df.to_csv('starwars1_processed.csv', encoding='utf-8', index=False)

something = 1