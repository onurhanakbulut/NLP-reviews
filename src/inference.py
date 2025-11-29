import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



MODEL_DIR = "../models/berturk_absa_v1"
ASPECTS_PATH = "../data/aspects.json"




def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = " ".join(text.split())
    return text



def load_aspects(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data= json.load(f)
    
    aspects = data["aspects"]
    id2name = {a["id"]: a["name"] for a in aspects}
    return aspects, id2name



def find_aspects(normalized_text: str, aspect_defs):
    
    found = []
    
    for aspect in aspect_defs:
        aid = aspect["id"]
        kws = aspect.get("keywords", [])
        for kw in kws:
            if kw in normalized_text:
                found.append(aid)
                break
            
    return list(set(found))






def build_model_input(aspect_id: str, comment: str, id2name: dict) -> str:
    
    aspect_name = id2name.get(aspect_id, aspect_id)
    return f"[ASPECT] {aspect_name} [SEP] {comment}"





print("Model ve tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval() #inference mod



id2label = {0: "negative", 1:"neutral", 2:"positive"}

aspects, id2name = load_aspects(ASPECTS_PATH)

print(f"{len(aspects)} aspect yüklendi: {[a['id'] for a in aspects]}")







def analyze_comment(comment: str):
    
    norm = normalize(comment)
    found_aspects = find_aspects(norm, aspects)
    
    
    if not found_aspects:
        print("\nBu yorumda aspect sözlüğüne göre hiçbir aspect bulunamadı.")
        return {
            "comment": comment,
            "normalized": norm,
            "results": []
            }
    
    
    
    results = []
    
    with torch.no_grad():
        for aid in found_aspects:
            model_input = build_model_input(aid, comment, id2name)
            
            
            encoded = tokenizer(
                model_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
                )
            
            
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            pred_label = id2label[pred_id]
            
# =============================================================================
#             print("LOGITS:", logits)
#             print("PROBS:", probs)
# =============================================================================

            
            
            
            results.append({
                "aspect_id": aid,
                "aspect_name": id2name.get(aid, aid),
                "sentiment": pred_label,
                "probs": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]),
                    "positive": float(probs[2])
                    
                    }
                })
            
            
    return {
        "comment": comment,
        "normalized": norm,
        "results": results
        }




if __name__ == "__main__":
    
    comment = input("Yorum: ")
    #comment = "Kumaşı çok güzel ama fiyatı pahalı, kargo da geç geldi."
    
    output = analyze_comment(comment)
    
    
    print("\nYORUM:")
    print(output["comment"])
    print("\nNORMALIZE EDİLMİŞ:")
    print(output["normalized"])

    print("\nBULUNAN ASPECT ve SENTIMENTLER:")
    for r in output["results"]:
        neg = r["probs"].get("negative", 0.0)
        neu = r["probs"].get("neutral", 0.0)
        pos = r["probs"].get("positive", 0.0)
    
        print(
            f"- {r['aspect_name']} ({r['aspect_id']}): {r['sentiment']} "
            f"(neg={neg:.2f}, neu={neu:.2f}, pos={pos:.2f})"
        )




