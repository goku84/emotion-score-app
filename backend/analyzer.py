import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import os

# --- Globals & Models ---
# Initialize models as None
nlp = None
embedder = None
emotion_tokenizer = None
emotion_xai_model = None
emotion_pipeline = None
sentiment_pipeline = None

def load_models():
    global nlp, embedder, emotion_tokenizer, emotion_xai_model, emotion_pipeline, sentiment_pipeline
    
    if nlp is not None:
        return

    print("Loading NLP models...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Initialize models
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Emotion models
    emotion_model_name = "SamLowe/roberta-base-go_emotions"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_xai_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name, output_attentions=True)
    emotion_xai_model.eval()

    emotion_pipeline = pipeline("text-classification", model=emotion_model_name, top_k=None)

    # Sentiment model
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    print("Models loaded.")


# --- Helper Functions ---

def emotion_with_attention(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = emotion_xai_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
    attentions = outputs.attentions
    return probs, attentions, inputs

def extract_emotion_words(attentions, inputs, top_k=6):
    last_layer = attentions[-1]
    avg_attention = last_layer.mean(dim=1)[0]
    token_scores = avg_attention.mean(dim=0).numpy()
    tokens = emotion_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_scores = token_scores / token_scores.max()
    word_importance = [(t, float(s)) for t, s in zip(tokens, token_scores) if t not in ["<s>", "</s>", "<pad>"]]
    word_importance.sort(key=lambda x: x[1], reverse=True)
    return word_importance[:top_k]

def extract_emotions(text):
    try:
        outputs = emotion_pipeline(text[:512])[0]
        return {o["label"]: o["score"] for o in outputs}
    except Exception:
        return {}

def extract_aspect_phrases(text):
    doc = nlp(text.lower())
    phrases = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:
            phrases.append(chunk.text.strip())
    return phrases

def emotion_penalty(emotion_score):
    return 1 - min(emotion_score, 0.9)

def compute_weighted_aspect_score(row):
    polarity = 1 if row["sentiment"] in ["LABEL_2", "POSITIVE"] else -1
    return polarity * row["sentiment_score"] * row["trust_score"] * emotion_penalty(row["emotion_score"])

def hierarchical_chunk_retrieval(sentences, top_k=6):
    filtered = [s for s in sentences if len(s.split()) > 4]
    if len(filtered) <= top_k:
        return filtered
    embeddings = embedder.encode(filtered)
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(embeddings, centroid).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [filtered[i] for i in top_indices]

def rag_fusion(aspect, sentences):
    queries = [f"user complaints about {aspect}", f"user satisfaction with {aspect}", f"issues reported in {aspect}"]
    all_selected = []
    for _ in queries:
        retrieved = hierarchical_chunk_retrieval(sentences)
        all_selected.extend(retrieved)
    return list(set(all_selected))

STOP_TERMS = {"everyone", "everything", "probably", "really", "things", "something", "someone", "loved"}

def multi_hop_reasoning(sentences):
    keywords = []
    for s in sentences:
        for word in s.lower().split():
            if word.isalpha() and len(word) > 4 and word not in STOP_TERMS:
                keywords.append(word)
    return [w for w, _ in Counter(keywords).most_common(5)]

def generate_aspect_summary(aspect, sentences):
    if not sentences:
        return f"No sufficient user feedback available for {aspect}."
    if aspect.lower() == "other":
        return "User feedback is diverse and does not consistently map to a specific product aspect."
    key_terms = multi_hop_reasoning(sentences)
    return f"Majority of users discuss {aspect.lower()} in terms of {', '.join(key_terms[:3])}, based on recurring feedback patterns."


def analyze_reviews(df):
    load_models()
    # Ensure DataFrame format
    if "Text" not in df.columns:
        # Try to find a text column
        possible_cols = ["text", "reviewText", "review_text", "content"]
        for col in possible_cols:
            if col in df.columns:
                df.rename(columns={col: "Text"}, inplace=True)
                break
        if "Text" not in df.columns:
             raise ValueError("Dataset must have a 'Text' column")

    # 1. Data Cleaning
    df = df.dropna(subset=["Text"])
    if "Summary" not in df.columns: df["Summary"] = ""
    df["Summary"] = df["Summary"].fillna("")
    if "ProfileName" not in df.columns: df["ProfileName"] = "anonymous"
    df["ProfileName"] = df["ProfileName"].fillna("anonymous")
    if "Rating" in df.columns:
        df = df[df["Rating"].between(1, 5)]

    # Deduplication
    cols_to_check = ["ProductId", "UserId", "Summary", "Text"]
    existing_cols = [c for c in cols_to_check if c in df.columns]
    if existing_cols:
        df = df.drop_duplicates(subset=existing_cols)

    # Time conversion
    if "Time" in df.columns:
        df["ReviewTime"] = pd.to_datetime(df["Time"], unit="s")
    else:
        # Create dummy time if missing needed for feature engineering
        df["ReviewTime"] = pd.to_datetime("now")

    df["review_year"] = df["ReviewTime"].dt.year
    df["review_month"] = df["ReviewTime"].dt.month
    df["review_date"] = df["ReviewTime"].dt.date
    
    # Feature Engineering
    if "UserId" in df.columns:
        df["user_review_count"] = df.groupby("UserId")["Text"].transform("count")
        df["user_avg_score"] = df.groupby("UserId")["Rating"].transform("mean") if "Rating" in df.columns else 0
        df["user_review_span_days"] = (df.groupby("UserId")["ReviewTime"].transform("max") - df.groupby("UserId")["ReviewTime"].transform("min")).dt.days.fillna(0)
    else:
        df["user_review_count"] = 1
        df["user_avg_score"] = 0
        df["user_review_span_days"] = 0

    if "ProductId" in df.columns:
        df["product_avg_score"] = df.groupby("ProductId")["Rating"].transform("mean") if "Rating" in df.columns else 0
        df["rating_deviation"] = abs(df["Rating"] - df["product_avg_score"]) if "Rating" in df.columns else 0
        df["reviews_per_day_product"] = df.groupby(["ProductId", "review_date"])["Text"].transform("count")
    else:
        df["product_avg_score"] = 0
        df["rating_deviation"] = 0
        df["reviews_per_day_product"] = 1

    # Text Embeddings
    text_embeddings = embedder.encode(df["Text"].tolist(), batch_size=64, show_progress_bar=True)
    
    # Duplicate Detection
    similarity_matrix = cosine_similarity(text_embeddings)
    df["near_duplicate_count"] = (similarity_matrix > 0.95).sum(axis=1) - 1

    # Text Features
    df["review_length"] = df["Text"].astype(str).apply(len)
    df["exclamation_count"] = df["Text"].astype(str).apply(lambda x: x.count("!"))
    df["capital_ratio"] = df["Text"].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    
    # Autoencoder
    feature_cols = ["review_length", "exclamation_count", "capital_ratio", "user_review_count", "user_avg_score", "user_review_span_days", "rating_deviation", "reviews_per_day_product", "near_duplicate_count"]
    # Handle missing columns if any
    for col in feature_cols:
        if col not in df.columns: df[col] = 0

    X_numeric = df[feature_cols].fillna(0)
    X_final = np.hstack([text_embeddings, X_numeric.values])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    autoencoder = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 128, 256), alpha=1e-4, max_iter=100, random_state=42)
    autoencoder.fit(X_scaled, X_scaled)
    reconstruction_error = ((X_scaled - autoencoder.predict(X_scaled)) ** 2).mean(axis=1)
    df["trust_score"] = 100 - (100 * (reconstruction_error / reconstruction_error.max()))

    # Fake/Genuine Label
    fake_threshold = df["trust_score"].quantile(0.20)
    df["review_label"] = df["trust_score"].apply(lambda x: "Fake" if x < fake_threshold else "Genuine")

    # Explanation
    def explain_fake_review(row):
        reasons = []
        if row["user_review_count"] > 50: reasons.append("Abnormally high number of reviews by user")
        if row["rating_deviation"] > 2: reasons.append("Rating deviates strongly from product average")
        if row["near_duplicate_count"] > 0: reasons.append("Review text similar to others")
        if row["review_length"] < 30: reasons.append("Review unusually short")
        if row["capital_ratio"] > 0.3: reasons.append("Excessive capital letters")
        if not reasons: reasons.append("Statistically anomalous review pattern")
        return "; ".join(reasons)

    df["fake_reason"] = df.apply(explain_fake_review, axis=1)

    # Emotions
    emotion_scores = df["Text"].apply(extract_emotions)
    emotion_df = pd.json_normalize(emotion_scores).fillna(0)
    df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)
    
    ekman_map = {
        "joy": ["joy"], "anger": ["anger"], "fear": ["fear"], "sadness": ["sadness"], "surprise": ["surprise"]
    }
    for ekman, labels in ekman_map.items():
        # Handle if labels are not in emotion_df (some emotions might not be present in smaller datasets)
        valid_labels = [l for l in labels if l in df.columns]
        if valid_labels:
            df[f"emotion_{ekman}"] = df[valid_labels].sum(axis=1)
        else:
            df[f"emotion_{ekman}"] = 0

    df["emotion_intensity"] = df[[f"emotion_{k}" for k in ekman_map.keys()]].max(axis=1)
    df["emotional_exaggeration"] = ((df["emotion_intensity"] > 0.85) & (df["exclamation_count"] > 2)).astype(int)

    # Aspect Mining
    SEED_ASPECTS = {
        "battery": ["battery", "backup", "charge", "power"],
        "camera": ["camera", "photo", "image", "video", "lens"],
        "display": ["screen", "display", "brightness", "resolution"],
        "performance": ["performance", "speed", "fast", "slow", "lag"],
        "heating": ["heat", "heating", "overheat"],
        "audio": ["audio", "sound", "speaker", "volume"],
        "price": ["price", "cost", "value", "money"],
        "build_quality": ["build", "quality", "material", "durable"]
    }

    all_phrases = []
    for review in df["Text"]:
        all_phrases.extend(extract_aspect_phrases(review))
    
    phrase_freq = Counter(all_phrases)
    aspect_candidates = [p for p, c in phrase_freq.items() if c >= 2]
    
    normalized_aspects = {}
    if aspect_candidates:
        aspect_embeddings =  embedder.encode(aspect_candidates) # Use the same embedder
        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2, metric="cosine", linkage="average")
        cluster_labels = cluster_model.fit_predict(aspect_embeddings)
        
        aspect_cluster_map = {}
        for phrase, label in zip(aspect_candidates, cluster_labels):
            aspect_cluster_map.setdefault(label, []).append(phrase)

        for label, phrases in aspect_cluster_map.items():
            mapped = []
            for phrase in phrases:
                found = False
                for aspect, keywords in SEED_ASPECTS.items():
                    for kw in keywords:
                        if kw in phrase:
                            mapped.append(aspect)
                            found = True
                            break
                    if found: break
                if not found: mapped.append("other")
            final_aspect = Counter(mapped).most_common(1)[0][0]
            normalized_aspects[label] = {"aspect": final_aspect, "phrases": phrases}

    def align_aspects(review):
        doc = nlp(review)
        aspect_units = []
        review_lower = review.lower()
        
        # Primary matching with phrases
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for data in normalized_aspects.values():
                for phrase in data["phrases"]:
                    if phrase in sent_text:
                        aspect_units.append({"aspect": data["aspect"], "text": sent.text})
                        break
        
        # Fallback to keywords
        if not aspect_units:
            for aspect, keywords in SEED_ASPECTS.items():
                if any(k in review_lower for k in keywords):
                    aspect_units.append({"aspect": aspect, "text": review})
                    break
        
        if not aspect_units:
            aspect_units.append({"aspect": "other", "text": review})
        return aspect_units

    df["aspect_units"] = df["Text"].apply(align_aspects)

    # Process Aspect Rows
    aspect_rows = []
    for idx, row in df.iterrows():
        trust = row.get("trust_score", 1.0)
        for unit in row["aspect_units"]:
            aspect_rows.append({
                "review_id": idx,  # Using index as temp ID
                "aspect": unit["aspect"],
                "text": unit["text"],
                "trust_score": trust
            })
    
    aspect_df = pd.DataFrame(aspect_rows)
    
    # Sentiment & Emotion on Aspects
    if not aspect_df.empty:
        texts = aspect_df["text"].tolist()
        # Batch process emotion
        emo_outputs = emotion_pipeline(texts, batch_size=16, truncation=True)
        aspect_df["emotion"] = [max(o, key=lambda x: x["score"])["label"] for o in emo_outputs]
        aspect_df["emotion_score"] = [max(o, key=lambda x: x["score"])["score"] for o in emo_outputs]
        
        # Batch process sentiment
        sent_outputs = sentiment_pipeline(texts, batch_size=16, truncation=True)
        aspect_df["sentiment"] = [o["label"] for o in sent_outputs]
        aspect_df["sentiment_score"] = [o["score"] for o in sent_outputs]

        aspect_df["weighted_aspect_score"] = aspect_df.apply(compute_weighted_aspect_score, axis=1)
    else:
        # Empty df structure
        aspect_df = pd.DataFrame(columns=["review_id", "aspect", "text", "trust_score", "emotion", "emotion_score", "sentiment", "sentiment_score", "weighted_aspect_score"])

    # Aspect Summary
    explainability_report = {}
    if not aspect_df.empty:
        for aspect, group in aspect_df.groupby("aspect"):
            explainability_report[aspect] = {
                "avg_weighted_score": group["weighted_aspect_score"].mean(),
                "dominant_sentiment": group["sentiment"].mode()[0] if not group["sentiment"].empty else "N/A",
                "dominant_emotion": group["emotion"].mode()[0] if not group["emotion"].empty else "N/A",
                "confidence_score": group["trust_score"].mean(),
                "evidence": group[["text", "trust_score", "emotion"]].head(5).to_dict("records")
            }
            
            # Generate RAG summary
            reviews_text = group["text"].tolist()
            fused_evidence = rag_fusion(aspect, reviews_text)
            explainability_report[aspect]["summary"] = generate_aspect_summary(aspect, fused_evidence)

    # Prepare Response
    # Convert df to records
    df["id"] = df.index
    reviews_data = df.fillna("").to_dict(orient="records")
    
    # Summary stats
    summary = {
        "total_reviews": len(df),
        "fake_reviews": len(df[df["review_label"] == "Fake"]),
        "genuine_reviews": len(df[df["review_label"] == "Genuine"]),
        "avg_trust_score": df["trust_score"].mean(),
    }

    return {
        "summary": summary,
        "reviews": reviews_data,
        "aspects": explainability_report
    }

if __name__ == "__main__":
    # Test run
    try:
        df = pd.read_csv("electronics_sample.csv")
        results = analyze_reviews(df)
        print("Analysis successful")
        print(results["summary"])
    except Exception as e:
        print(f"Error: {e}")
