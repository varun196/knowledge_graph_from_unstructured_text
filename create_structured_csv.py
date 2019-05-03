import pickle
import pandas as pd
import os


def main():
    with open('named_entities.pickle','rb') as f:
        entities = pickle.load(f)
    
    entity_set = set(entities.keys())
    final_list = []

    curr_dir = os.getcwd()
    df = pd.read_csv(curr_dir +"stanford-openie/corpus/input.txt-out.csv")

    triplet = []
    for i,j in df.iterrows():
        j[0] = j[0].strip()
        if j[0] in entity_set:
            added = False
            for entity in j[2]:
                if entity in entity_set:
                    _ = (entities[j[0]], str(j[0]), str(j[1]) ,entity, str(j[2]) )
                    triplet.append(_)
                    added = True
            if not added:
                _ = (entities[j[0]], str(j[0]), str(j[1]) ,'OBJ', str(j[2]) )
                triplet.append(_)

    processed_pd = pd.DataFrame(triplet)
    processed_pd.to_csv('processed.csv', encoding='utf-8', index=False)

    print("File processed and saved")

if __name__ == '__main__':
    main()