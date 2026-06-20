# F1 Decision DNA: Quantifying Pit Stop Strategy

This quantitative pipeline extracts each Formula 1 constructor's implicit pit decision function from publicly available lap time telemetry—what we define as their **Decision DNA**.

Using the 2021 Formula 1 season (the final year of an eight-year regulatory era), we engineered a two-stage deterministic model to isolate driver behavior from car physics.

## Methodology
* **Gaussian Mixture Model (GMM):** Identifies distinct, unsupervised physical tyre degradation regimes per constructor.
* **Logistic Regression (Statsmodels):** Extracts specific behavioral coefficients quantifying how each team and driver reacts to isolated stimuli under race conditions.

## The Metrics
* **β₁ (Degradation Sensitivity):** Likelihood to pit based strictly on physical car degradation.
* **β₂ (Position Sensitivity):** Likelihood to pit based on immediate track position threats.
* **AUC:** The predictability score of the strategy (closer to 0.50 = higher strategic opacity / global game theory execution).

---

## Key Findings: Team Level

The midfield displays highly predictable, reactive strategies (AUC > 0.55), tied heavily to mathematical tyre drop-offs (positive β₁) and immediate track position (negative β₂).

| Constructor | β₁ | β₂ | AUC | Predictability |
| :--- | :---: | :---: | :---: | :--- |
| **Ferrari** | 0.0330 | -0.2341 | 0.6993 | Highly Reactive / Predictable |
| **Alpine** | 0.0413 | -0.0994 | 0.6184 | Reactive |
| **Haas F1** | 0.0304 | -0.0650 | 0.6055 | Reactive |
| **Alfa Romeo** | 0.0290 | -0.2797 | 0.5721 | Mid-Tier Reactive |
| **AlphaTauri** | 0.0292 | -0.0718 | 0.5634 | Mid-Tier Reactive |
| **Williams** | 0.0293 | -0.1419 | 0.5569 | Mid-Tier Reactive |
| **McLaren** | 0.0365 | -0.1225 | 0.5554 | Mid-Tier Reactive |
| **Red Bull Racing**| 0.0214 | -0.0137 | 0.5458 | Championship / Opaque |
| **Aston Martin** | 0.0245 | -0.0295 | 0.5363 | Opaque |
| **Mercedes** | 0.0189 | -0.0661 | 0.5244 | Championship / Highly Opaque |

---

## Key Findings: Driver Level (The Bimodality Hypothesis)

A mathematical decoupling occurs at the championship level. Top constructors assign reactive, physics-bound strategies to secondary drivers, freeing their primary drivers to execute statistically opaque, global game-theoretic strategies.

| Team | Driver | β₁ | β₂ | AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Mercedes** | HAM | 0.0145 | -0.0809 | 0.5952 |
| **Mercedes** | BOT | 0.0252 | -0.0872 | 0.5092 |
| **Red Bull** | PER | 0.0256 | -0.0367 | 0.5987 |
| **Red Bull** | VER | 0.0159 | +0.0242 | 0.4966 |
