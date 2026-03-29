# PEOPLECERT_assessment
# Career Coach Recommender Mini-Project

## Overview

This project implements a simplified recommendation engine for a Career Coach AI assistant.
The goal is to recommend the **next best certificates** for a learner based on:

* user role
* user skills
* career goal
* purchase history
* certificate prerequisites
* global interaction popularity

The final system combines:

* **content-based filtering** using TF-IDF and cosine similarity
* **co-occurrence-based recommendations** from purchase history
* **popularity scoring** from user interaction signals

It also exposes recommendations through a **FastAPI endpoint**.

---

## Project Structure

```text
├── ML_Engineer_Candidate _test.ipynb
├── app.py
├── dockerfile
├── catalog.csv
├── users.csv
├── purchases.csv
├── signals.csv
├── requirements.txt
└── README.md

```

---

## Methodology

### 1. Data Exploration

The datasets were explored to identify:

* most popular certificates
* common user roles
* common learning pathways
* interaction trends

### 2. Candidate Generation

Two candidate generation methods were implemented:

#### a. Content-Based Filtering

A user profile text was created from:

* role
* skills
* goal

A certificate text was created from:

* certificate name
* certificate skills
* short description

Then **TF-IDF** and **cosine similarity** were used to measure relevance between users and certificates.

#### b. Co-occurrence Filtering

Using purchase history, I counted how often certificates were purchased together across users.
This captures pathway patterns such as:

> users who purchased certificate A often also purchased certificate B

### 3. Ranking

The final recommendation score is a weighted combination of:

* **content score**
* **co-occurrence score**
* **popularity score**

Formula:

```text
final_score =
0.5 * content_score +
0.30 * cooccurrence_score +
0.2 * popularity_score
```

### 4. Business Rules

The recommender applies the following constraints:

* exclude certificates already purchased by the user
* enforce prerequisites
* return top-5 recommendations

### 5. Explainability

Each recommendation includes a short reason such as:

* matches your skills
* frequently taken with your previous certificates
* popular among similar learners

---

## Popularity Score

To model global engagement, interaction events from `signals.csv` were weighted as follows:

* impression = 1
* click = 3
* add_to_cart = 5
* purchase = 8

These weighted events were aggregated per certificate and normalized to produce a popularity score between 0 and 1.

---



## KMeans Discussion

KMeans clustering was considered as a possible way to group similar certificates.
However, it was not used in the final model because:

* the dataset is relatively small
* clustering does not directly model user-specific preferences
* it does not capture learning pathways or prerequisites

Instead, a hybrid recommender combining content similarity, co-occurrence, and interaction-based popularity was chosen.

---

## Requirements Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Running the API

Start the FastAPI server with:

```bash
uvicorn app:app --reload
```


---

## API Endpoint

### Get recommendations

```http
GET /recommend?user_id=<USER_ID>&top_k=5
```

### Example

```bash
curl "http://127.0.0.1:8000/recommend?user_id=12345&top_k=5"
```


## Interactive API Docs

FastAPI automatically provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

## Response example 

[
  {
    "cert_id": "MSPF",
    "name": "MSP Foundation",
    "reason": "Matches your skills in stakeholder engagement, benefits management",
    "score": 0.2525
  },
  {
    "cert_id": "DevOpsF",
    "name": "DevOps Foundation",
    "reason": "Popular among learners with similar interests",
    "score": 0.1797
  },
  {
    "cert_id": "PRINCE2F",
    "name": "PRINCE2 Foundation",
    "reason": "Popular among learners with similar interests",
    "score": 0.1746
  },
  {
    "cert_id": "ITILF",
    "name": "ITIL Foundation",
    "reason": "Matches your skills in itsm",
    "score": 0.1619
  },
  {
    "cert_id": "ScrumF",
    "name": "Scrum Foundation",
    "reason": "Popular among learners with similar interests",
    "score": 0.1368
  }
]

## Running the dockerfile 
First create a docker image
```bash
docker  build -t peoplecert_assessment .
```
Then run the image 
```bash
docker  run -p 8000:8000 peoplecert_assessment
```
Then open
```text
http://localhost:8000/docs
```

## Future Improvements

Potential next steps include:

* evaluation using **Recall@5** 
* semantic embeddings for better text matching
* real-time feedback logging
* Unit tests


---

## Conclusion

This project demonstrates a practical and explainable hybrid recommendation system for certification pathways.
It combines user profile matching, historical purchase patterns, and global engagement signals to recommend the next best certificates for learners.
