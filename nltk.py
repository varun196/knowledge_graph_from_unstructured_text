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

class CustomNER:
    
    def ner(self,doc):
        tagged = self.assign_pos_tags(doc)
        print(tagged)

    def assign_pos_tags(self,doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged

doc = "The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall."

# Custom ER
custom_ner = CustomNER()
custom_ner.ner(doc)

# Stanford NER
stanford_ner = StanfordNER()
tagged = stanford_ner.ner(doc)
#print(tagged)

# Spacy NER
spacy_ner = SpacyNER()
tagged = spacy_ner.ner(doc)
#print(tagged)