# 13 — Recommendation System: MovieLens

Difficulty: 🟠 Medium-Hard | Topic: Recommender Systems

## 🎯 Project Goal
Build and honestly evaluate a movie recommender on MovieLens 100k, progressing from popularity baseline to collaborative filtering, and confront the evaluation question most tutorials skip: *what does "good" even mean for a recommender?*

## 📊 Dataset + Evaluation Metric
- **Dataset:** MovieLens 100k — 100,000 ratings (1–5) from 943 users on 1,682 movies, plus genres. Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip (the `u.data` and `u.item` files are what you need).
- **Evaluation metrics:** RMSE for rating prediction, AND Precision@10 / Recall@10 for top-N recommendation (define a rating ≥4 as "relevant"). Use a temporal or leave-last-k-out split per user — random splits leak.

## 🏁 Success Criteria
- Three approaches implemented and compared: (1) popularity/mean baselines, (2) item-based collaborative filtering with cosine similarity, (3) matrix factorization (e.g., via SVD/ALS — implementing it with numpy is encouraged but a library is acceptable)
- Rating-prediction RMSE < 0.95; top-10 Precision@10 reported for all three
- Cold-start discussion backed by numbers: how do the models perform for users with <10 ratings?
- Qualitative spot check: show the top-10 recommendations for 3 distinct users and argue whether they're plausible

Relevant techniques (look them up yourself): user-item matrix, cosine similarity, neighborhood-based CF, matrix factorization / latent factors, Precision@k, popularity bias.
