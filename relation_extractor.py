import os
import subprocess
import pandas as pd

def Stanford_Relation_Extractor():

    
    print('Relation Extraction Started')
    files = ['input.txt']
    
    for f in files:
        
        print(f)
        os.chdir('/Users/rajat/Desktop/Knowledge_Graph/stanford-openie')

        p = subprocess.Popen(['./process_large_corpus.sh','corpus/'+f,'corpus/' + f + '-out.csv'], stdout=subprocess.PIPE)

        output, err = p.communicate()

        

    print('Relation Extraction Completed')


if __name__ == '__main__':
    Stanford_Relation_Extractor()
