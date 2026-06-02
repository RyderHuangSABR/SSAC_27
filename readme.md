CopyF1 Decision DNA
Extracting Behavioral Decision Coefficients from Formula 1 Pit Stop Data
Target: MIT Sloan Sports Analytics Conference 2027
Research Question: What do pit stop decisions reveal about how each F1 team thinks?

Overview
This pipeline extracts each Formula 1 constructor's implicit pit decision function
from publicly available lap time telemetry — what we call their Decision DNA.
Using the 2021 Formula 1 season (the final year of an eight-year regulatory era),
we apply a two-stage model:

Gaussian Mixture Model — identifies tyre degradation regimes per constructor
Logistic Regression — extracts behavioral coefficients quantifying each team's
sensitivity to degradation (β1) vs track position (β2)


β1 = Degradation Sensitivity
β2 = Position Sensitivity
AUC = Predictability score (closer to 0.5 = more strategically opaque)


Key Findings — Team Level
Mercedes | β1: 0.0189 | β2: -0.0661 | AUC: 0.5244
Red Bull Racing | β1: 0.0214 | β2: -0.0137 | AUC: 0.5458
Ferrari | β1: 0.0330 | β2: -0.2341 | AUC: 0.6993
McLaren | β1: 0.0365 | β2: -0.1225 | AUC: 0.5554
Alpine | β1: 0.0413 | β2: -0.0994 | AUC: 0.6184
AlphaTauri | β1: 0.0292 | β2: -0.0718 | AUC: 0.5634
Aston Martin | β1: 0.0245 | β2: -0.0295 | AUC: 0.5363
Williams | β1: 0.0293 | β2: -0.1419 | AUC: 0.5569
Alfa Romeo Racing | β1: 0.0290 | β2: -0.2797 | AUC: 0.5721
Haas F1 Team | β1: 0.0304 | β2: -0.0650 | AUC: 0.6055

Key Findings — Driver Level (Bimodality Hypothesis Test)
Mercedes — HAM | β1: 0.0145 | β2: -0.0809 | AUC: 0.5952
Mercedes — BOT | β1: 0.0252 | β2: -0.0872 | AUC: 0.5092
Red Bull — VER | β1: 0.0159 | β2: +0.0242 | AUC: 0.4966
Red Bull — PER | β1: 0.0256 | β2: -0.0367 | AUC: 0.5987

Notable: Max Verstappen is the only driver with a positive β2 (+0.0242)
and a below-random AUC (0.4966) — quantifying an aggressive strategic
personality previously documented only anecdotally.


Requirements
pip install fastf1 pandas numpy scikit-learn joblib

Usage
pythonpython f1_decision_dna.py

Data Source
All data sourced via FastF1

Author
Independent researcher
MIT Sloan Sports Analytics Conference 2027 submission
