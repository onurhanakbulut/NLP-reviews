import json
import pandas as pd


with open("../data/aspects.json", "r", encoding="utf-8") as f:
    aspects = json.load(f)["aspects"]
    
    
df = pd.read_csv("../data/clean/rew_norm_token.csv")
    
    
    
def find_aspects(text, aspect_defs):
    found = []
    
    
    for aspect in aspect_defs:
        aid = aspect["id"]
        kws = aspect["keywords"]
        
        
        for kw in kws:
            if kw in text:
                found.append(aid)
                break
            
    return list(set(found))



def map_starts_to_sentiment(stars):
    
    if stars in [1, 2]:
        return 'negative'
    
    elif stars == 3:
        return 'neutral'
    
    elif stars in [4, 5]:
        return 'positive'
    
    else:
        return None         ##eksikse
    
    

    

df['sentiment_comment'] = df['stars'].apply(map_starts_to_sentiment)
    



df["aspect_ids"] = df["comment_norm"].apply(lambda t: find_aspects(t, aspects))

df_aspects = df.explode("aspect_ids", ignore_index=True)        #sütundaki değerler 1'den fazlaysa birkaç satırda gösteriyor
df_aspects = df_aspects[df_aspects["aspect_ids"].notna()]   #null clean


df_aspects['sentiment_weak'] = df_aspects['sentiment_comment']



#save
df.to_csv('../data/processed/aspect_sentiment_added_all_rew.csv')
df_aspects.to_csv('../data/processed/df_aspects_exploded.csv')




















