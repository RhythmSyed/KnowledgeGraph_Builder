import os
os.environ['CLASSPATH'] = '/Users/rhythmsyed/Desktop/GTRI/entitylink/miniepy/target/minie-0.0.1-SNAPSHOT.jar'

from miniepy import *

# Instantiate minie
minie = MinIE()

# Sentence to extract triples from
sentence = "The Joker believes that the hero Batman was not actually born in foggy Gotham City."

# Get proposition triples
triples = [p.triple for p in minie.get_propositions(sentence)]

print("Original sentence:")
print('\t{}'.format(sentence))

print("Extracted triples:")
for t in triples:
    print("\t{}".format(t))