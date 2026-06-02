F1 Decision DNA
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


Key Findings
Teamβ1 (Degradation)β2 (Position)AUCMercedes0.0189-0.06610.5244Red Bull Racing0.0214-0.01370.5458Ferrari0.0330-0.23410.6993McLaren0.0365-0.12250.5554Alpine0.0413-0.09940.6184AlphaTauri0.0292-0.07180.5634Aston Martin0.0245-0.02950.5363Williams0.0293-0.14190.5569Alfa Romeo0.0290-0.27970.5721Haas0.0304-0.06500.6055
Notable finding: Max Verstappen is the only driver with a positive β2 (+0.0242)
and a below-random AUC (0.4966) — quantifying an aggressive strategic personality
previously documented only anecdotally.

Requirements
pip install fastf1 pandas numpy scikit-learn joblib

Usage
pythonpython f1_decision_dna.py
The pipeline will:

Download 2021 F1 season data via FastF1 API
Clean and normalize lap times
Fit GMM degradation models per constructor
Extract Decision DNA coefficients via logistic regression
Run bimodality hypothesis test for Mercedes and Red Bull
Export all models to ./models/


Output
Exported models per constructor:

{team}_scaler.joblib — StandardScaler for tyre physics features
{team}_gmm.joblib — Gaussian Mixture Model for degradation regimes
{team}_behavior.joblib — Logistic regression Decision DNA model


Data Source
All data sourced via FastF1 —
an open-source Python library pulling from the official F1 timing API.

Author
Independent researcher
MIT Sloan Sports Analytics Conference 2027 submission
