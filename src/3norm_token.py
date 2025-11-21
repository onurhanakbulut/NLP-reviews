import pandas as pd
import re



df = pd.read_csv('../data/clean/all_rew_clean.csv')


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    
    return text


def tokenize(text):
    return text.split()






yorumlar_norm = []
yorumlar_token = []

for yorum in df['comment']:
    norm = normalize(yorum)
    token = tokenize(norm)
    
    yorumlar_norm.append(norm)
    yorumlar_token.append(token)
    
    
    
#print(yorumlar)

df['comment_norm'] = pd.DataFrame({"norm": yorumlar_norm})
df['comment_token'] = pd.DataFrame({"token": yorumlar_token})

df = df.rename(columns={ df.columns[0]: "index_id"})
df.to_csv('../data/clean/rew_norm_token.csv', index=False, encoding="utf-8")

    
    