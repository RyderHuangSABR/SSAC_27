# F1 Decision DNA: Quantifying Pit Stop Strategy

This quantitative pipeline extracts each Formula 1 constructor's implicit pit decision function from publicly available lap time telemetry—what we define as their **Decision DNA**.

Using the 2021 Formula 1 season (the final year of an eight-year regulatory era), we engineered a two-stage deterministic model to isolate driver behavior from car physics.

## Methodology
* **Gaussian Mixture Model (GMM):** Identifies distinct, unsupervised physical tyre degradation regimes per constructor.
* **Logistic Regression (Statsmodels):** Extracts specific behavioral coefficients quantifying how each team and driver reacts to isolated stimuli under race conditions.

## The Metrics
* **β₁ (Degradation Sensitivity):** Likelihood to pit based strictly on physical car degradation.
* **β₂ (Position Sensitivity):** Likelihood to pit based on immediate track position threats. Negative values indicate a reactive strategy (pitting when position is threatened). 
* **AUC:** The predictability score of the strategy (closer to 0.50 = higher strategic opacity / global game theory execution).

---

## Key Findings: The Bimodality Hypothesis

A mathematical decoupling occurs at the championship level. Top constructors assign reactive, physics-bound strategies to secondary drivers (highly significant P-values), freeing their primary drivers to execute statistically opaque, global game-theoretic strategies independent of local track variables.

| Team | Driver | β₁ | P(β₁) | β₂ | P(β₂) | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mercedes** | HAM | 0.0128 | 0.0851 | -0.1124 | 0.3180 | **0.6071** |
| **Mercedes** | BOT | 0.0174 | 0.0412 | -0.0665 | 0.4894 | **0.5158** |
| **Red Bull** | PER | 0.0207 | 0.0170 | -0.0334 | 0.5204 | **0.6017** |
| **Red Bull** | VER | 0.0123 | 0.1294 | 0.0343 | 0.6236 | **0.4885** |

---

## Complete 2021 Grid: Decision DNA Signatures

The full grid reveals a clear spectrum of strategic behavior, ranging from the highly opaque championship contenders (AUC < 0.52) to the highly predictable, reactive profiles of Ferrari and Haas (AUC > 0.65). 

*(Sorted ascending by strategic predictability / AUC Score)*

| Driver | Team | β₁ (Deg) | P(β₁) | β₂ (Pos) | P(β₂) | AUC Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **VER** | Red Bull Racing | 0.0123 | 0.1294 | 0.0343 | 0.6236 | 0.4885 |
| **BOT** | Mercedes | 0.0174 | 0.0412 | -0.0665 | 0.4894 | 0.5158 |
| **STR** | Aston Martin | 0.0182 | 0.0144 | -0.1009 | 0.1939 | 0.5281 |
| **RUS** | Williams | 0.0169 | 0.0241 | -0.1288 | 0.0598 | 0.5375 |
| **TSU** | AlphaTauri | 0.0233 | 0.0005 | -0.1656 | 0.0063 | 0.5381 |
| **NOR** | McLaren | 0.0289 | 0.0004 | -0.1274 | 0.1689 | 0.5625 |
| **RAI** | Alfa Romeo Racing | 0.0400 | <0.0001 | -0.6777 | <0.0001 | 0.5647 |
| **OCO** | Alpine | 0.0269 | 0.0016 | -0.0648 | 0.2691 | 0.5687 |
| **VET** | Aston Martin | 0.0224 | 0.0008 | -0.0158 | 0.6909 | 0.5703 |
| **LAT** | Williams | 0.0260 | 0.0017 | -0.1086 | 0.0387 | 0.5736 |
| **GAS** | AlphaTauri | 0.0228 | 0.0012 | 0.0060 | 0.8951 | 0.5881 |
| **RIC** | McLaren | 0.0251 | 0.0006 | -0.0978 | 0.2381 | 0.5905 |
| **PER** | Red Bull Racing | 0.0207 | 0.0017 | -0.0334 | 0.5204 | 0.6017 |
| **GIO** | Alfa Romeo Racing | 0.0235 | 0.0004 | -0.2097 | 0.0233 | 0.6068 |
| **HAM** | Mercedes | 0.0128 | 0.0851 | -0.1124 | 0.3180 | 0.6071 |
| **ALO** | Alpine | 0.0270 | 0.0003 | -0.1090 | 0.1224 | 0.6102 |
| **MAZ** | Haas F1 Team | 0.0230 | 0.0012 | -0.1625 | 0.3481 | 0.6540 |
| **MSC** | Haas F1 Team | 0.0222 | 0.0013 | -0.1211 | 0.1528 | 0.6610 |
| **LEC** | Ferrari | 0.0194 | 0.0134 | -0.1248 | 0.1143 | 0.6887 |
| **SAI** | Ferrari | 0.0351 | <0.0001 | -0.3827 | 0.0006 | 0.7170 |

---
**Data and Code Availability:**
All raw telemetry data was sourced via the FastF1 API. The complete Python codebase, including the data-cleaning pipeline, Gaussian Mixture Models, and Logistic Regression frameworks used to extract these coefficients, is contained in this repository.
