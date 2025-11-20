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



df["aspect_ids"] = df["comment_norm"].apply(lambda t: find_aspects(t, aspects))

df_aspects = df.explode("aspect_ids", ignore_index=True)        #sütundaki değerler 1'den fazlaysa birkaç satırda gösteriyor
df_aspects = df_aspects[df_aspects["aspect_ids"].notna()]


