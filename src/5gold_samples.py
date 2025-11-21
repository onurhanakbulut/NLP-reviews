import pandas as pd



df = pd.read_csv('../data/processed/aspect_sentiment_added_all_rew.csv')

df_aspects = pd.read_csv('../data/processed/df_aspects_exploded.csv')





df_train = df_aspects[['comment_norm', 'aspect_ids', 'sentiment_weak']]

df_train = df_train.rename(columns={
    'comment_norm': 'text',
    'aspect_ids': 'aspect',
    'sentiment_weak': 'label'
    })



#df_train = df_train.dropna(subset=['text', 'aspect', 'label'])


label2id = {            #dictionary
    "negative": 0,
    "neutral":1,
    "positive": 2
    }


# =============================================================================
# id2label = {}                
# for key, value in label2id.items():
#     id2label[value] = key
# =============================================================================


id2label = {value: key for key, value in label2id.items()}      #dictionary comprehension


df_train['label_id'] = df_train['label'].map(label2id)






from sklearn.model_selection import train_test_split


train_df , temp_df = train_test_split(df_train, test_size=0.2, random_state=11, stratify=df_train['label_id'])

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=11, stratify=temp_df['label_id'])


####GOLD TEST SET

gold_sample = (test_df.groupby(['aspect', 'label'], group_keys= False).apply(lambda g: g.sample(min(len(g), 50), random_state=11)))





gold_sample['gold_label'] = gold_sample['label']
gold_sample.to_csv('../data/goldsample/gold_sample.csv', index=False, encoding=('utf-8'))

####################AFTER GOLD LABEL

gold_df = pd.read_csv('../data/goldsample/gold_sample_123.csv')


gold_df = gold_df.dropna()

gold_df['gold_label_id'] = gold_df['gold_label'].map(label2id)


######













