import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationSystem:
    def __init__(self, output_dir="outputs/recommendation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rating_matrix = None
        self.predicted_scores = None

    def create_sample_ratings(self, user_count=80, item_count=50, density=0.2):
        np.random.seed(42)

        rows = []

        user_groups = np.random.randint(0, 4, user_count)
        item_groups = np.random.randint(0, 4, item_count)

        for user_id in range(user_count):
            for item_id in range(item_count):
                if np.random.rand() > density:
                    continue

                base_score = 3.0
                group_bonus = 1.2 if user_groups[user_id] == item_groups[item_id] else 0
                noise = np.random.normal(0, 0.7)

                rating = base_score + group_bonus + noise
                rating = np.clip(rating, 1, 5)

                rows.append({
                    "user_id": f"user_{user_id}",
                    "item_id": f"item_{item_id}",
                    "rating": round(float(rating), 1)
                })

        ratings = pd.DataFrame(rows)
        return ratings

    def build_rating_matrix(self, ratings):
        matrix = ratings.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            aggfunc="mean"
        )

        self.rating_matrix = matrix
        return matrix

    def train_user_based_model(self):
        filled_matrix = self.rating_matrix.fillna(0)

        similarity = cosine_similarity(filled_matrix)
        similarity_df = pd.DataFrame(
            similarity,
            index=self.rating_matrix.index,
            columns=self.rating_matrix.index
        )

        np.fill_diagonal(similarity_df.values, 0)

        weighted_sum = similarity_df.dot(filled_matrix)
        similarity_sum = np.abs(similarity_df).sum(axis=1)

        similarity_sum = similarity_sum.replace(0, np.nan)

        self.predicted_scores = weighted_sum.div(similarity_sum, axis=0).fillna(0)

    def recommend(self, user_id, top_n=5):
        if user_id not in self.predicted_scores.index:
            return []

        user_scores = self.predicted_scores.loc[user_id].copy()

        already_rated_items = self.rating_matrix.loc[user_id].dropna().index
        user_scores = user_scores.drop(index=already_rated_items, errors="ignore")

        top_items = user_scores.sort_values(ascending=False).head(top_n)

        recommendations = [
            {
                "user_id": user_id,
                "item_id": item_id,
                "predicted_score": round(score, 4)
            }
            for item_id, score in top_items.items()
        ]

        return recommendations

    def recommend_for_all_users(self, top_n=5):
        result = []

        for user_id in self.rating_matrix.index:
            result.extend(self.recommend(user_id, top_n=top_n))

        return pd.DataFrame(result)

    def evaluate_simple_hit_rate(self, ratings, top_n=5, threshold=4.0):
        high_rating_items = ratings[ratings["rating"] >= threshold]

        hit_count = 0
        total_count = 0

        for user_id in high_rating_items["user_id"].unique():
            actual_items = set(
                high_rating_items[
                    high_rating_items["user_id"] == user_id
                ]["item_id"]
            )

            recommended_items = set(
                item["item_id"] for item in self.recommend(user_id, top_n=top_n)
            )

            if not recommended_items:
                continue

            hit_count += len(actual_items & recommended_items)
            total_count += len(recommended_items)

        if total_count == 0:
            return 0

        return hit_count / total_count

    def run(self):
        ratings = self.create_sample_ratings()
        ratings.to_csv(
            self.output_dir / "ratings.csv",
            index=False,
            encoding="utf-8-sig"
        )

        self.build_rating_matrix(ratings)
        self.train_user_based_model()

        recommendations = self.recommend_for_all_users(top_n=5)

        recommendations.to_csv(
            self.output_dir / "recommendations.csv",
            index=False,
            encoding="utf-8-sig"
        )

        hit_rate = self.evaluate_simple_hit_rate(ratings)

        pd.DataFrame([{
            "hit_rate": hit_rate
        }]).to_csv(
            self.output_dir / "evaluation.csv",
            index=False
        )

        print("추천 시스템 실행 완료")
        print("평점 데이터 수:", len(ratings))
        print("추천 결과 수:", len(recommendations))
        print("Hit Rate:", round(hit_rate, 4))
        print(recommendations.head(10))


if __name__ == "__main__":
    recommender = RecommendationSystem()
    recommender.run()
