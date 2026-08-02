import json
import math
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

PATH = "data/v5c_results/v6_global_ranking_3x20.json"
SIMULATIONS = 2000
SEED = 42

with open(PATH, "r") as f:
    result = json.load(f)

details = result["details"]

def probability_vector(detail):
    probabilities = detail["probabilities"]
    return np.asarray(
        [
            float(probabilities.get(str(n), probabilities.get(n)))
            for n in range(1, 50)
        ],
        dtype=float,
    )

def target_vector(detail):
    actual = set(int(n) for n in detail["actual_numbers"])
    return np.asarray(
        [1 if n in actual else 0 for n in range(1, 50)],
        dtype=int,
    )

total_hits = sum(int(detail["hits"]) for detail in details)
hits_at_5 = total_hits / len(details)

auc_values = []
ap_values = []

score_vectors = []
target_vectors = []

for detail in details:
    y = target_vector(detail)
    scores = probability_vector(detail)

    target_vectors.append(y)
    score_vectors.append(scores)

    auc_values.append(roc_auc_score(y, scores))
    ap_values.append(average_precision_score(y, scores))

mean_auc = float(np.mean(auc_values))
mean_ap = float(np.mean(ap_values))

rng = np.random.default_rng(SEED)

permuted_hits = np.empty(SIMULATIONS)
permuted_auc = np.empty(SIMULATIONS)
permuted_ap = np.empty(SIMULATIONS)

for simulation in range(SIMULATIONS):
    simulation_hits = 0
    simulation_auc = []
    simulation_ap = []

    for scores, y in zip(score_vectors, target_vectors):
        shuffled_scores = rng.permutation(scores)

        top5 = np.argsort(shuffled_scores)[-5:]
        simulation_hits += int(y[top5].sum())

        simulation_auc.append(
            roc_auc_score(y, shuffled_scores)
        )

        simulation_ap.append(
            average_precision_score(y, shuffled_scores)
        )

    permuted_hits[simulation] = simulation_hits / len(details)
    permuted_auc[simulation] = np.mean(simulation_auc)
    permuted_ap[simulation] = np.mean(simulation_ap)

def p_value(distribution, observed):
    return float(
        (np.sum(distribution >= observed) + 1)
        / (len(distribution) + 1)
    )

print("=" * 72)
print("PREDIXA V6 3x20 STATISTICAL TEST")
print("=" * 72)

print("Hits@5 observed :", round(hits_at_5, 6))
print("Hits@5 random   :", round(float(np.mean(permuted_hits)), 6))
print("Hits@5 p-value  :", round(p_value(permuted_hits, hits_at_5), 6))

print()

print("ROC-AUC observed:", round(mean_auc, 6))
print("ROC-AUC random  :", round(float(np.mean(permuted_auc)), 6))
print("ROC-AUC p-value :", round(p_value(permuted_auc, mean_auc), 6))

print()

print("AP observed     :", round(mean_ap, 6))
print("AP random       :", round(float(np.mean(permuted_ap)), 6))
print("AP p-value      :", round(p_value(permuted_ap, mean_ap), 6))
