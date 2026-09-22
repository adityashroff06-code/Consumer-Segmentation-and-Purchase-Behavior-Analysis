# Consumer Segmentation & Purchase Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-K--Means%20%7C%20Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityashroff06-code/Consumer-Segmentation-and-Purchase-Behavior-Analysis/blob/main/BWS_FInal_Project.ipynb)

End-to-end market segmentation of **600 households** from the classic CRISA bath-soap panel study: K-Means clustering on purchase behavior, cluster profiling, and a classification layer that tests how well *demographics alone* can predict a household's behavioral segment.

## Business problem

A consumer-goods marketer wants to move beyond demographic targeting: can households be segmented by *how they actually buy* (brand loyalty, volume, price sensitivity, promo responsiveness) — and can those segments then be reached using demographic data that's easy to collect?

## Dataset

`BathSoapHousehold.csv` — 600 households from the CRISA (Indian market research) bath-soap panel, a standard case study in data-mining courses (Table 21.8 of *Data Mining for Business Analytics*). Three variable groups:

- **Demographics** — socioeconomic class (SEC), eating habits (FEH), mother tongue (MT), sex, age, education, household size, children, TV availability, Affluence Index
- **Purchase behavior** — number of brands, brand runs, total volume, transactions, value, volume per transaction, average price
- **Basis of purchase** — % volume with/without promotions, price-category and brand-code proportions

## Approach

1. **EDA & data quality** — variable review, missing-value audit (none found)
2. **Feature engineering** — select purchase-behavior + basis-of-purchase features; standardize with `StandardScaler`
3. **Cluster count selection** — elbow method (inertia) cross-checked with silhouette scores → **k = 4**
4. **K-Means segmentation** — assign each household a cluster label
5. **Cluster profiling** — per-cluster means and visual comparison (bar charts, heatmap) to translate clusters into marketing personas
6. **Demographic classification** — train classifiers (incl. Random Forest) to predict the behavioral cluster from demographics only
7. **Feature importance** — identify which demographic signals matter most

## Key findings

- Purchase behavior separates cleanly into **4 segments** (e.g., loyal low-volume vs. promo-driven brand switchers), while
- **Demographics predict those segments only weakly** — per-class F1 scores are modest, an honest negative result that argues against pure demographic targeting
- **Affluence Index is by far the strongest demographic signal** (importance ≈ 0.25), followed by household size (0.15), education (0.11), and socioeconomic class (0.11)
- Practical implication: target by behavioral data where available; when only demographics are available, affluence-based proxies are the best fallback

## Project structure

```
├── BWS_FInal_Project.ipynb   # Full analysis: EDA → K-Means → profiling → classification
├── BathSoapHousehold.csv     # 600-household CRISA panel dataset
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting started

**Option A — Colab (zero setup):** click the *Open in Colab* badge above and upload `BathSoapHousehold.csv` when prompted.

**Option B — local:**

```bash
git clone https://github.com/adityashroff06-code/Consumer-Segmentation-and-Purchase-Behavior-Analysis.git
cd Consumer-Segmentation-and-Purchase-Behavior-Analysis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook BWS_FInal_Project.ipynb
```

> The first cell uses `google.colab.files.upload()`; when running locally, skip it and load the CSV directly with `pd.read_csv("BathSoapHousehold.csv")`.

## Tech stack

pandas · NumPy · scikit-learn (K-Means, silhouette, Random Forest) · matplotlib · seaborn

## License

Released under the [MIT License](LICENSE).

## Author

**Aditya Shroff** — [GitHub](https://github.com/adityashroff06-code) · [LinkedIn](https://www.linkedin.com/in/aditya-shroff-8033a31b0)
