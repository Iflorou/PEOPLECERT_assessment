from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

app = FastAPI(title="Career Coach Recommender API")

# -----------------------------
# Load data
# -----------------------------
cat_df = pd.read_csv("catalog.csv")
purch_df = pd.read_csv("purchases.csv", dtype={"user_id": str})
signal_df = pd.read_csv("signals.csv", dtype={"user_id": str})
user_df = pd.read_csv("users.csv", dtype={"user_id": str})
cat_df["prerequisites"] = cat_df["prerequisites"].fillna("")

# -----------------------------
# Build content-based features
# -----------------------------
user_df["profile_text"] = (
    user_df["role"].fillna("") + " " +
    user_df["skills"].fillna("") + " " +
    user_df["goal"].fillna("")
)

cat_df["cert_text"] = (
    cat_df["name"].fillna("") + " " +
    cat_df["skills"].fillna("") + " " +
    cat_df["short_desc"].fillna("")
)

all_text = pd.concat([user_df["profile_text"], cat_df["cert_text"]])

vectorizer = TfidfVectorizer()
vectorizer.fit(all_text)

user_vectors = vectorizer.transform(user_df["profile_text"])
cert_vectors = vectorizer.transform(cat_df["cert_text"])

similarity_matrix = cosine_similarity(user_vectors, cert_vectors)

content_score_df = pd.DataFrame(
    similarity_matrix,
    index=user_df["user_id"],
    columns=cat_df["cert_id"]
)

# -----------------------------
# Build co-occurrence
# -----------------------------
user_cert_map = purch_df.groupby("user_id")["cert_id"].apply(list)
co_occurrence = defaultdict(lambda: defaultdict(int))

for certs in user_cert_map.values:
    for i in certs:
        for j in certs:
            if i != j:
                co_occurrence[i][j] += 1

max_cooccurrence = max(
    [count for related in co_occurrence.values() for count in related.values()],
    default=1
)

# -----------------------------
# Build popularity
# -----------------------------
event_counts = signal_df.groupby(["cert_id", "event"]).size().unstack(fill_value=0)

for col in ["impression", "click", "add_to_cart", "purchase"]:
    if col not in event_counts.columns:
        event_counts[col] = 0

event_counts["popularity_score"] = (
    1 * event_counts["impression"] +
    3 * event_counts["click"] +
    5 * event_counts["add_to_cart"] +
    8 * event_counts["purchase"]
)

event_counts["popularity_score"] = (
    event_counts["popularity_score"] / event_counts["popularity_score"].max()
)

popularity_score_dict = event_counts["popularity_score"].to_dict()

# -----------------------------
# Helper functions
# -----------------------------
def get_user_purchases(user_id):
    return set(
        purch_df[purch_df["user_id"] == user_id]["cert_id"].astype(str).tolist()
    )

def has_prerequisites(user_id, cert_id):
    user_purchases = get_user_purchases(user_id)

    cert_match = cat_df.loc[cat_df["cert_id"] == cert_id, "prerequisites"]
    if cert_match.empty:
        return False

    prereq_value = cert_match.iloc[0]

    if pd.isna(prereq_value) or str(prereq_value).strip() == "":
        return True

    required_certs = {x.strip() for x in str(prereq_value).split("|") if x.strip()}
    return required_certs.issubset(user_purchases)

def get_cooccurrence_score(user_id, candidate_cert_id):
    user_purchases = get_user_purchases(user_id)

    score = 0
    for purchased_cert in user_purchases:
        score += co_occurrence[purchased_cert].get(candidate_cert_id, 0)

    return score

def get_normalized_cooccurrence_score(user_id, candidate_cert_id):
    raw_score = get_cooccurrence_score(user_id, candidate_cert_id)
    return raw_score / max_cooccurrence if max_cooccurrence else 0

def recommend_hybrid(user_id, top_k=5):
    user_purchases = get_user_purchases(user_id)
    recommendations = []

    for cert_id in cat_df["cert_id"]:
        if cert_id in user_purchases:
            continue

        if not has_prerequisites(user_id, cert_id):
            continue

        content_score = content_score_df.loc[user_id, cert_id]
        cooccurrence_score = get_normalized_cooccurrence_score(user_id, cert_id)
        popularity_score = popularity_score_dict.get(cert_id, 0)

        final_score = (
            0.5 * content_score +
            0.3 * cooccurrence_score +
            0.2 * popularity_score
        )

        cert_row = cat_df[cat_df["cert_id"] == cert_id].iloc[0]

        recommendations.append({
            "cert_id": cert_id,
            "name": cert_row["name"],
            "content_score": round(float(content_score), 4),
            "cooccurrence_score": round(float(cooccurrence_score), 4),
            "popularity_score": round(float(popularity_score), 4),
            "final_score": round(float(final_score), 4)
        })

    recommendations = sorted(recommendations, key=lambda x: x["final_score"], reverse=True)
    return pd.DataFrame(recommendations[:top_k])

def build_reason(user_id, cert_id, content_score, cooccurrence_score, popularity_score):
    user_row = user_df[user_df["user_id"] == user_id].iloc[0]
    cert_row = cat_df[cat_df["cert_id"] == cert_id].iloc[0]

    user_skills = set(str(user_row["skills"]).lower().split("|")) if pd.notna(user_row["skills"]) else set()
    cert_skills = set(str(cert_row["skills"]).lower().split("|")) if pd.notna(cert_row["skills"]) else set()

    overlap = user_skills.intersection(cert_skills)

    if overlap:
        return f"Matches your skills in {', '.join(list(overlap)[:2])}"
    if cooccurrence_score > 0:
        return "Frequently taken next by similar learners"
    if popularity_score > 0:
        return "Popular among learners with similar interests"
    return "Relevant to your profile and career goal"

def recommend_for_user(user_id, top_k=5):
    if user_id not in set(user_df["user_id"].tolist()):
        raise HTTPException(status_code=404, detail="User not found")

    recs_df = recommend_hybrid(user_id, top_k=top_k).copy()

    results = []
    for _, row in recs_df.iterrows():
        results.append({
            "cert_id": row["cert_id"],
            "name": row["name"],
            "reason": build_reason(
                user_id,
                row["cert_id"],
                row["content_score"],
                row["cooccurrence_score"],
                row["popularity_score"]
            ),
            "score": float(row["final_score"])
        })

    return results

# -----------------------------
# API endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Career Coach Recommender API is running"}

@app.get("/recommend")
def recommend(user_id: str, top_k: int = 5):
    return recommend_for_user(user_id, top_k)