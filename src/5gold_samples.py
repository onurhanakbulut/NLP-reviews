import pandas as pd
import json
import numpy as np


df = pd.read_csv('../data/processed/aspect_sentiment_added_all_rew.csv')

df_aspects = pd.read_csv('../data/processed/df_aspects_exploded.csv')



with open("../data/aspects.json", "r", encoding='utf-8') as f:
    aspect_defs = json.load(f)["aspects"]
    
id2name = {a["id"]: a["name"] for a in aspect_defs}




def build_model_input(row):
    aspect_id = row["aspect"]
    aspect_name = id2name.get(aspect_id)
    text = row["text"]
    return f"[ASPECT] {aspect_name} [SEP] {text}"














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
##egitim = train_df
##val = val_df
##test = gold_df




MODEL_NAME = "dbmdz/bert-base-turkish-cased"

###[ASPECT] Kumaş Kalitesi [SEP] Kumaşı güzel ama fiyatı biraz pahalı.


train_df["model_input"] = train_df.apply(build_model_input, axis=1)
val_df["model_input"] = val_df.apply(build_model_input, axis= 1)
gold_df["model_input"] = gold_df.apply(build_model_input, axis=1)







#HUGGINGFACE DATASET TRANSFROM
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[["model_input", "label_id"]])
val_dataset = Dataset.from_pandas(val_df[["model_input", "label_id"]])
gold_dataset = Dataset.from_pandas(gold_df[["model_input", "gold_label_id"]])




from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 256


def tokenize_fn(batch):
    return tokenizer(
        batch["model_input"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
        )
####tokenizer input_ids, attention_mask gibi kolonlar oluşturur.


train_tokenized = train_dataset.map(tokenize_fn, batched=True)
val_tokenized = val_dataset.map(tokenize_fn, batched=True)
gold_tokenized = gold_dataset.map(tokenize_fn, batched=True)


#HF labels adı ile input beklediği için labels yaptık
train_tokenized = train_tokenized.rename_column("label_id", "labels")
val_tokenized   = val_tokenized.rename_column("label_id", "labels")
gold_tokenized  = gold_tokenized.rename_column("gold_label_id", "labels")


#gereksiz pandas kolonları temizlendi
cols_to_remove = [c for c in train_tokenized.column_names if c not in ["input_ids", "attention_mask", "labels"]]
train_tokenized = train_tokenized.remove_columns([c for c in cols_to_remove if c in train_tokenized.column_names])
val_tokenized   = val_tokenized.remove_columns([c for c in cols_to_remove if c in val_tokenized.column_names])
gold_tokenized  = gold_tokenized.remove_columns([c for c in cols_to_remove if c in gold_tokenized.column_names])





#MODEL


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
    )



def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1
        }







training_args = TrainingArguments(
    output_dir= "berturk_absa",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50
    )


trainer = Trainer(
    model=model,
    args= training_args,
    train_dataset = train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )




trainer.train()



gold_results = trainer.evaluate(gold_tokenized)
gold_results


preds_raw = trainer.predict(gold_tokenized)
y_true = preds_raw.label_ids
y_pred = np.argmax(preds_raw.predictions, axis=-1)



from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label.keys())]))




trainer.model.save_pretrained("berturk_absa_v1")
tokenizer.save_pretrained("berturk_absa_v1")

























