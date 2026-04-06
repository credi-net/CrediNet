# CrediBench Leaderboard

Benchmark results for domain credibility prediction. Both tasks use a **0–1 scale; higher is better**.

> **Want to appear here?** Run your method on CrediBench and [open an issue](https://github.com/credi-net/CrediNet/issues) with your results, code, and contact information.

---

## Tasks

- [Binary Classification](#binary-classification)
- [Regression](#regression)

---

## Binary Classification

### Domain credibility binary classification

##### Accuracy and F1 Scores on test set. **Higher is better.**

Predict whether a domain is credible (1) or not (0).
The main evaluation metrics is **Accuracy**.

| Rank | Method | Acc. | F1 | Team | Year |
| :--: | ------ | :---: | :----: | ---- | ---- |
| **1.** | **MLP (Graph + Text) *(Ours)*** | **83.6 $\pm$ 0.03** | **83.2** | **CrediNet team** | **2025** | 
| 2. | GAT w/ Text Embeddings *(Ours)* | 76.1 $\pm$ 0.36 | 75.2 | CrediNet team | 2025 | 
| 3. | GAT w/ Random Initialization *(Ours)* | 68.9 $\pm$ 0.18 | 69.7 | CrediNet team | 2025 | 
| 3. | Text Embedding-based *(Ours)* | 63.2 $\pm$ 0.02 | 60.5 | CrediNet team | 2025 | 
| 4. | LightGBM | 56.0 | 70.7 | Kadkhoda et al. | 2025 |
| 5. | LLM-URL + Web Search | 54.35 | 63.03 | Yang et al. *(enhanced)* | 2025 | 
| 6. | Constant | 53.0 | 69.1 | N/A | N/A | 
| 7. | LLM-URL | 52.84 | 63.9 | Yang et al. | 2025 | 
| 8. | SEO-based GNN | 50.0 | 66.7 | Carragher et al. | 2025| 



---

## Regression

### Domain credibility scoring (regression)

##### Mean and Max. Absolute Error on test set. **Higher is better.**

Predict a continuous credibility score in **[0, 1]**.
The main evaluation metric is **Mean Absolute Error (MAE)**.

| Rank | Method | MAE | Max(AE) | Team | Year |
| :--: | ------ | :---: | :----: | ---- | ---- |
| **1.** | **MLP (Graph + Text) *(Ours)*** | **0.112 $\pm$ 0.001** | **0.544** | **CrediNet team** | **2025** | 
| 2. | GAT w/ Text Embeddings *(Ours)* | 0.114 $\pm$ 0.001 | 0.477 | CrediNet team | 2025 | 
| 3. | GCN w/ Text Embeddings *(Ours)* | 0.114 $\pm$ 0.002 | 0.849 | CrediNet team | 2025 | 
| 4. | LightGBM | 0.145 | 0.630 | Kadkhoda et al. | 2025 |
| 5. | LLM-URL + Web Search | 0.158 | 0.719 | Yang et al. *(enhanced)* | 2025 | 
| 6. | LLM-URL | 0.162 | 0.765 | Yang et al. | 2025 | 
| 7. | Mean | 0.167 | 0.546 | N/A | N/A | 
| 8. | SEO-based GNN | 0.428 | 0.956 | Carragher et al. | 2025| 

