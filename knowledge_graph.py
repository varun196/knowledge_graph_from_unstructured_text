import nltk
import sys
import pickle
# For Spacy:
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
# For custom ER:
import tkinter
import re
# For Coreference resolution
import json
from stanfordcorenlp import StanfordCoreNLP

class StanfordNER:
    def __init__(self,loc):
        self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(loc+'/classifiers/english.all.3class.distsim.crf.ser.gz',
        loc+'/stanford-ner.jar')

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

def nltk_ner(doc):
    print("\nusing NLTK for NER")
    # NLTK NER
    nltk_ner = NltkNER()
    tagged = nltk_ner.ner(doc)
    print("\n\nTagged: \n\n")
    pprint(tagged)
    print("\n\nTree: \n\n ")
    for leaves in tagged:
        print(leaves)
        #leaves.draw()
    print("\n")

def stanford_ner(doc):
    print("Provide (relative/absolute) path to stanford ner package.\n Press carriage return to use './stanford-ner-2018-10-16' as path:") 
    loc = input()
    print("using Stanford for NER (may take a while):  \n\n\n")
    if(loc == ''):
        loc = "./stanford-ner-2018-10-16"
    stanford_ner = StanfordNER(loc)
    tagged = stanford_ner.ner(doc)
    print(tagged)
    print("\n")

def spacy_ner(doc):
    print("\nusing Spacy for NER\n")
    spacy_ner = SpacyNER()
    tagged = spacy_ner.ner(doc)
    print(tagged)
    print("\n")

class CoreferenceResolver:
    def generate_coreferences(self,doc,stanford_core_nlp_path):
        '''
        pickles results object to coref_res.p
        the result has the following structure:
        dict of dict of lists of dicts:  { { [ {} ] } }  -- We are interested in the 'corefs' key { [ { } ] }-- Each list has all coreferences to a given pronoun.
        '''
        nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=False)
        props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
        annotated = nlp.annotate(doc, properties=props)
        print("\nannotated\n\n",annotated,"\n\n")
        result = json.loads(annotated)
        # Dump coreferences to a file
        pickle.dump(result,open( "coref_res.p", "wb" ))
        # Close server to release memory
        nlp.close()

    def unpickle(self):
        result = pickle.load(open( "coref_res.p", "rb" ))
        result = result['corefs']
        #print(result)
        for key in result:
            None
            print(result[key]) 
            print("\n")
        return result
    
    def resolve_coreferences(self):
        result = pickle.load(open( "coref_res.p", "rb" ))
        result = result['corefs']
        None        

def resolve_coreferences(doc,stanford_core_nlp_path):
    doc = "The Godfather Vito Corleone is the head of the Corleone mafia family in New York. He is at the event of his daughter's wedding. Michael, Vito's youngest son and a decorated WW II Marine is also present at the wedding. Michael seems to be uninterested in being a part of the family business. Vito is a powerful man, and is kind to all those who give him respect but is ruthless against those who do not. But when a powerful and treacherous rival wants to sell drugs and needs the Don's influence for the same, Vito refuses to do it. What follows is a clash between Vito's fading old values and the new ways which may cause Michael to do the thing he was most reluctant in doing and wage a mob war against all the other mafia families which could tear the Corleone family apart."
    coref = CoreferenceResolver()
    coref.generate_coreferences(doc,stanford_core_nlp_path)
    coref.unpickle()

def main():
    if len(sys.argv) == 1:
        print("Usage:   python3 knowledge_graph.py <nltk/stanford/spacy> [<nltk/stanford/spacy> <nltk/stanford/spacy>]")
    
    stanford_core_nlp_path = input("Provide (relative/absolute) path to stanford core nlp package.\n Press carriage return to use './stanford-corenlp-full-2018-10-05' as path:")
    if(stanford_core_nlp_path == ''):
        stanford_core_nlp_path = "./stanford-corenlp-full-2018-10-05"

    doc1 = "The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta."

    doc = "The Godfather Vito Corleone is the head of the Corleone mafia family in New York. He is at the event of his daughter's wedding. Michael, Vito's youngest son and a decorated WW II Marine is also present at the wedding. Michael seems to be uninterested in being a part of the family business. Vito is a powerful man, and is kind to all those who give him respect but is ruthless against those who do not. But when a powerful and treacherous rival wants to sell drugs and needs the Don's influence for the same, Vito refuses to do it. What follows is a clash between Vito's fading old values and the new ways which may cause Michael to do the thing he was most reluctant in doing and wage a mob war against all the other mafia families which could tear the Corleone family apart."

    for i in range(1,len(sys.argv)):
        if(sys.argv[i] == "nltk"):
            nltk_ner(doc)
        elif(sys.argv[i]=="stanford"):
            stanford_ner(doc)
        elif(sys.argv[i]=="spacy"):
            spacy_ner(doc)
    
    resolve_coreferences(doc,stanford_core_nlp_path)

main()