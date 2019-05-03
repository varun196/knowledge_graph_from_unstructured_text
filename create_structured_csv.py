import pickle
import pandas as pd
import os
import glob


def main():

    pickles = []
    for file in glob.glob(os.getcwd() + "/data/output/ner/*.pickle"):
        pickles.append(file)

    for file in pickles:
        with open(file,'rb') as f:
            entities = pickle.load(f)
    
        entity_set = set(entities.keys())
        final_list = []
        curr_dir = os.getcwd()
        df = pd.read_csv(curr_dir +"/stanford-openie/corpus/input.txt-out.csv")

        triplet = set()
        for i,j in df.iterrows():
            j[0] = j[0].strip()
            if j[0] in entity_set:
                added = False
                e2_sentence = j[2].split(' ')
                for entity in e2_sentence:
                    if entity in entity_set:
                        _ = (entities[j[0]], str(j[0]), str(j[1]) ,entities[entity], str(j[2]) )
                        triplet.add(_)
                        added = True
                if not added:
                    _ = (entities[j[0]], str(j[0]), str(j[1]) ,'O', str(j[2]) )
                    triplet.add(_)

        processed_pd = pd.DataFrame(list(triplet),columns=['Type','Entity 1','Relationship','Type', 'Entity2'])
        processed_pd.to_csv('./data/result/processed' + file.split("/")[-1].split(".")[0] + '.csv', encoding='utf-8', index=False)

        print("Processed " + file.split("/")[-1])

    print("Files processed and saved")

if __name__ == '__main__':
    main()