import nltk
# For Spacy:
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
# For custom ER:
import tkinter
import re

class StanfordNER:
    stanford_ner_tagger = nltk.tag.StanfordNERTagger('/home/varun/Downloads/Temp/ADBI_capstone/Project/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/home/varun/Downloads/Temp/ADBI_capstone/Project/stanford-ner-2018-10-16/stanford-ner.jar')

    def ner(self,doc):
        sentences = nltk.sent_tokenize(doc)
        result = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = self.stanford_ner_tagger.tag(words)
            result.append(tagged)
        return result
    
class SpacyNER:
    def ner(self,doc):    
        nlp = en_core_web_sm.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]

class NltkNER:
    def ner(self,doc):
        pos_tagged = self.assign_pos_tags(doc)
        #chunks = self.split_into_chunks(pos_tagged)
        result = []
        for sent in pos_tagged:
            result.append(nltk.ne_chunk(sent))
        return result

    def assign_pos_tags(self,doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged
    
    def split_into_chunks(self,sentences):
        # This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (DT) or possessive pronoun (PRP$) followed by any number of adjectives (JJ/JJR/JJS) and then any number of nouns (NN/NNS/NNP/NNPS) {dictator/NN Kim/NNP Jong/NNP Un/NNP}. Using this grammar, we create a chunk parser.
        grammar = "NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        chunks = []
        for sent in sentences:
            chunks.append(cp.parse(sent))
        return chunks

doc = "The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta."

print("NLTK: \n")
# NLTK NER
nltk_ner = NltkNER()
tagged = nltk_ner.ner(doc)
print("Tagged: \n")
pprint(tagged)
print("Tree: \n")
for leaves in tagged:
    print(leaves)
    leaves.draw()

print("\n")

# Stanford NER 

print("Stanford: \n") 
stanford_ner = StanfordNER()
tagged = stanford_ner.ner(doc)
print(tagged)

print("\n")
# Spacy NER

print("Spacy: \n")
spacy_ner = SpacyNER()
tagged = spacy_ner.ner(doc)
print(tagged)
