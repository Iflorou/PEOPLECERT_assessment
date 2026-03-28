from collections import defaultdict
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Career Coach Recommender API")


# -----------------------------
# Load data
# -----------------------------
cat_df = pd.read_csv("catalog.csv")
purch_df = pd.read_csv("purchases.csv")
signal_df = pd.read_csv("signals.csv")
user_df = pd.read_csv("users.csv")

# Ensure purchase_date is parsed for possible evaluation / ordering
if "purchase_date" in purch_df.columns:
    purch_df["purchase_date"] = pd.to_datetime(purch_df["purchase_date"], errors="coerce")

# -----------------------------
# Basic cleaning
# -----------------------------
cat_df["prerequisites"] = cat_df["prerequisites"].fillna("")
cat_df["skills"] = cat_df["skills"].fillna("")
cat_df["short_desc"] = cat_df["short_desc"].fillna("")
cat_df["name"] = cat_df["name"].fillna("")

user_df["role"] = user_df["role"].fillna("")
user_df["skills"] = user_df["skills"].fillna("")
user_df["goal"] = user_df["goal"].fillna("")

# Keep user_id / cert_id consistent as strings
cat_df["cert_id"] = cat_df["cert_id"].astype(str)
purch_df["cert_id"] = purch_df["cert_id"].astype(str)
purch_df["user_id"] = purch_df["user_id"].astype(str)
signal_df["cert_id"] = signal_df["cert_id"].astype(str)
signal_df["user_id"] = signal_df["user_id"].astype(str)
user_df["user_id"] = user_df["user_id"].astype(str)

# -----------------------------
# Content-based preparation
# -----------------------------
user_df["profile_text"] = (
    user_df["role"].astype(str) + " " +
    user_df["skills"].astype(str).str.replace("|", " ", regex=False) + " " +
    user_df["goal"].astype(str)
)

cat_df["cert_text"] = (
    cat_df["name"].astype(str) + " " +
    cat_df["skills"].astype(str).str.replace("|", " ", regex=False) + " " +
    cat_df["short_desc"].astype(str)
)

all_text = pd.concat([user_df["profile_text"], cat_df["cert_text"]], axis=0)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(all_text)

n_users = len(user_df)
user_tfidf = tfidf_matrix[:n_users]
cert_tfidf = tfidf_matrix[n_users:]

similarity_matrix = cosine_similarity(user_tfidf, cert_tfidf)

content_score_df = pd.DataFrame(
    similarity_matrix,
    index=user_df["user_id"],
    columns=cat_df["cert_id"]
)

# -----------------------------
# Co-occurrence preparation
# -----------------------------
user_cert_map = purch_df.groupby("user_id")["cert_id"].apply(list)

co_occurrence = defaultdict(lambda: defaultdict(int))

for certs in user_cert_map.values:
    unique_certs = list(set(certs))
    for i in unique_certs:
        for j in unique_certs:
            if i != j:
                co_occurrence[i][j] += 1

max_cooccurrence = max(
    [count for related in co_occurrence.values() for count in related.values()],
    default=1
)

# -----------------------------
# Popularity score preparation
# -----------------------------
event_counts = signal_df.groupby(["cert_id", "event"]).size().unstack(fill_value=0)

for col in ["impression", "click", "add_to_cart", "purchase"]:
    if col not in event_counts.columns:
        event_counts[col] = 0

event_counts["signal_score"] = (
    1 * event_counts["impression"] +
    3 * event_counts["click"] +
    5 * event_counts["add_to_cart"] +
    8 * event_counts["purchase"]
)

if event_counts["signal_score"].max() > 0:
    event_counts["signal_score"] = event_counts["signal_score"] / event_counts["signal_score"].max()

signal_score_dict = event_counts["signal_score"].to_dict()

purchase_counts = purch_df["cert_id"].value_counts()
if len(purchase_counts) > 0 and purchase_counts.max() > 0:
    purchase_score_dict = (purchase_counts / purchase_counts.max()).to_dict()
else:
    purchase_score_dict = {}

combined_popularity_score_dict = {}
for cert_id in cat_df["cert_id"]:
    signal_score = signal_score_dict.get(cert_id, 0)
    purchase_score = purchase_score_dict.get(cert_id, 0)
    combined_popularity_score_dict[cert_id] = 0.6 * signal_score + 0.4 * purchase_score


# -----------------------------
# Helper functions
# -----------------------------
def get_user_purchases(user_id: str) -> set:
    return set(
        purch_df[purch_df["user_id"] == user_id]["cert_id"].tolist()
    )


def has_prerequisites(user_id: str, cert_id: str) -> bool:
    user_purchases = get_user_purchases(user_id)
    prereq_value = cat_df.loc[cat_df["cert_id"] == cert_id, "prerequisites"].iloc[0]

    if pd.isna(prereq_value) or str(prereq_value).strip() == "":
        return True

    required_certs = set(str(prereq_value).split("|"))
    return required_certs.issubset(user_purchases)


def get_normalized_cooccurrence_score(user_id: str, candidate_cert_id: str) -> float:
    user_purchases = get_user_purchases(user_id)

    score = 0
    for purchased_cert in user_purchases:
        score += co_occurrence[purchased_cert].get(candidate_cert_id, 0)

    return score / max_cooccurrence if max_cooccurrence > 0 else 0.0


def generate_reason(user_id: str, cert_id: str) -> str:
    user_row = user_df[user_df["user_id"] == user_id].iloc[0]
    cert_row = cat_df[cat_df["cert_id"] == cert_id].iloc[0]

    reasons: List[str] = []

    user_skills = set(str(user_row["skills"]).split("|")) if str(user_row["skills"]).strip() else set()
    cert_skills = set(str(cert_row["skills"]).split("|")) if str(cert_row["skills"]).strip() else set()
    overlap = user_skills.intersection(cert_skills)

    if overlap:
        reasons.append(f"Matches your skills in {', '.join(list(overlap)[:2])}")

    if get_normalized_cooccurrence_score(user_id, cert_id) > 0:
        reasons.append("Frequently taken with your previous certificates")

    if not reasons:
        reasons.append("Popular among learners with similar interests")

    return ". ".join(reasons[:2]) + "."


def recommend_hybrid_with_reason(user_id: str, top_k: int = 5) -> list:
    user_purchases = get_user_purchases(user_id)
    recommendations = []

    for cert_id in cat_df["cert_id"]:
        # Exclude already purchased certificates
        if cert_id in user_purchases:
            continue

        # Enforce prerequisites
        if not has_prerequisites(user_id, cert_id):
            continue

        # Scores
        content_score = float(content_score_df.loc[user_id, cert_id])
        cooccurrence_score = float(get_normalized_cooccurrence_score(user_id, cert_id))
        popularity_score = float(combined_popularity_score_dict.get(cert_id, 0))

        final_score = (
            0.45 * content_score +
            0.30 * cooccurrence_score +
            0.25 * popularity_score
        )

        cert_row = cat_df[cat_df["cert_id"] == cert_id].iloc[0]

        recommendations.append({
            "cert_id": cert_id,
            "name": cert_row["name"],
            "reason": generate_reason(user_id, cert_id),
            "score": round(final_score, 4)
        })

    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)
    return recommendations[:top_k]


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(user_id: str, top_k: int = 5):
    if user_id not in set(user_df["user_id"]):
        raise HTTPException(status_code=404, detail="User not found")

    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")

    return recommend_hybrid_with_reason(user_id=user_id, top_k=top_k)