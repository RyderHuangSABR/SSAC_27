import os
import fastf1
import joblib
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
 
# ─── SETUP ────────────────────────────────────────────────────────────────────
# Paths work locally AND on Kaggle
if os.path.exists('/kaggle/working'):
    CACHE_DIR  = '/kaggle/working/f1_cache'
    EXPORT_DIR = '/kaggle/working/models'
else:
    CACHE_DIR  = './f1_cache'
    EXPORT_DIR = './models'
 
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA EXTRACTION & FUEL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
 
def load_f1_data(year: int, round_number: int) -> pd.DataFrame:
    print(f"--> Downloading AWS Data for {year} Round {round_number}...")
    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=False)
 
    laps = session.laps
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
 
    df = laps[[
        'Driver', 'Team', 'LapNumber', 'LapTime_s',
        'Stint', 'Compound', 'TyreLife', 'Position',
        'IsAccurate', 'PitInTime'
    ]].copy()
    df = df.rename(columns={'Driver': 'DriverId'})
    df['Round'] = round_number
 
    # Flag pit laps before any filtering
    df['IsPitLap'] = ~df['PitInTime'].isnull()
 
    # Target: did this driver pit on the NEXT lap?
    df = df.sort_values(['DriverId', 'LapNumber'])
    df['Did_Pit'] = (
        df.groupby('DriverId')['IsPitLap']
          .shift(-1).fillna(False).astype(int)
    )
 
    # Fuel correction (~0.03s per kg, 110kg start)
    total_laps = df['LapNumber'].max()
    df['FuelRemaining_kg'] = 110.0 * (1.0 - (df['LapNumber'] / total_laps))
    df['NormalizedPace_s'] = df['LapTime_s'] - (df['FuelRemaining_kg'] * 0.03)
 
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL ANOMALY FILTERING (Option 1 + Option 3)
# ══════════════════════════════════════════════════════════════════════════════
 
def filter_global_anomalies(df: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
    """
    Fleet-wide residual anomaly detection.
    If >65% of drivers show anomalous pace on the same lap,
    it's a track condition (SC/VSC/yellow), not individual deg.
    """
    racing_laps = df[df['IsAccurate'] == True].copy()
    if racing_laps.empty:
        return df
 
    fleet_medians = racing_laps.groupby('LapNumber')['NormalizedPace_s'].transform('median')
    racing_laps['DeltaToFleet'] = racing_laps['NormalizedPace_s'] - fleet_medians
 
    anomalous_laps = racing_laps.groupby('LapNumber').filter(
        lambda x: (x['DeltaToFleet'] > 1.5).mean() > threshold
    )['LapNumber'].unique()
 
    return df[~df['LapNumber'].isin(anomalous_laps)].copy()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. TEAM PIPELINE: GMM + LOGISTIC REGRESSION + EXPORT
# ══════════════════════════════════════════════════════════════════════════════
 
def process_team(train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 target_team: str,
                 driver_level: bool = False):
    """
    Full pipeline for one team:
      Stage A - Physics engine: StandardScaler + GMM on TyreLife/NormalizedPace
      Stage B - Behavioral data prep: DegradationPercent per stint
      Stage C - Decision DNA: LogisticRegression on Degradation + Position
      Stage D - Export models as .joblib files
      Stage E - Evaluate + print results
 
    If driver_level=True, also splits by individual driver to test
    the bimodality hypothesis (HAM vs BOT, VER vs PER).
    """
    print(f"\n{'='*70}")
    print(f" PIPELINE: {target_team.upper()}")
    print(f"{'='*70}")
 
    team_train = train_df[train_df['Team'] == target_team].copy()
    team_test  = test_df[test_df['Team']  == target_team].copy()
 
    # ── Stage A: Physics Engine ───────────────────────────────────────────────
    train_racing = team_train[
        (team_train['IsAccurate'] == True) &
        (~team_train['IsPitLap'])
    ].dropna(subset=['NormalizedPace_s'])
 
    if len(train_racing) < 50:
        print(f"  Skipping: insufficient data ({len(train_racing)} laps)")
        return
 
    scaler = StandardScaler()
    X_phys_train = train_racing[['TyreLife', 'NormalizedPace_s']]
    X_scaled     = scaler.fit_transform(X_phys_train)
 
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
 
    # Identify cliff regime = component with highest mean NormalizedPace_s
    train_racing = train_racing.copy()
    train_racing['Regime'] = gmm.predict(X_scaled)
    regime_paces  = train_racing.groupby('Regime')['NormalizedPace_s'].mean()
    cliff_regime  = regime_paces.idxmax()
 
    # ── Stage B: Behavioral Data Prep ─────────────────────────────────────────
    def prepare_behavioral(df: pd.DataFrame) -> pd.DataFrame:
        racing = df[
            (df['IsAccurate'] == True) &
            (~df['IsPitLap'])
        ].dropna(subset=['NormalizedPace_s']).copy()
 
        if racing.empty:
            return pd.DataFrame()
 
        # Apply pre-trained scaler + GMM
        racing['Regime'] = gmm.predict(
            scaler.transform(racing[['TyreLife', 'NormalizedPace_s']])
        )
 
        # Baseline pace per stint = mean pace of non-cliff laps
        baselines = []
        for (driver, stint), group in racing.groupby(['DriverId', 'Stint']):
            non_cliff = group[group['Regime'] != cliff_regime]
            base = (non_cliff['NormalizedPace_s'].mean()
                    if not non_cliff.empty
                    else group['NormalizedPace_s'].mean())
            baselines.append({
                'DriverId':    driver,
                'Stint':       stint,
                'BaselinePace': base
            })
 
        baseline_df = pd.DataFrame(baselines)
        if baseline_df.empty:
            return pd.DataFrame()
 
        merged = df.merge(baseline_df, on=['DriverId', 'Stint'], how='left')
        merged = merged.dropna(subset=['BaselinePace', 'NormalizedPace_s'])
        merged['DegradationPercent'] = (
            (merged['NormalizedPace_s'] - merged['BaselinePace'])
            / merged['BaselinePace'] * 100
        )
 
        return merged[[
            'Round', 'DriverId', 'LapNumber',
            'DegradationPercent', 'Position', 'Did_Pit'
        ]].dropna()
 
    train_beh = prepare_behavioral(team_train)
    test_beh  = prepare_behavioral(team_test)
 
    if (len(train_beh[train_beh['Did_Pit'] == 1]) < 3 or
            len(test_beh[test_beh['Did_Pit'] == 1]) < 1):
        print(f"  Skipping: insufficient pit examples in train/test")
        return
 
    # ── Stage C: Decision DNA (Team Level) ────────────────────────────────────
    def fit_and_evaluate(X_tr, y_tr, X_te, y_te, label):
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1]
        acc   = accuracy_score(y_te, preds)
        try:
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = np.nan
 
        print(f"\n  [{label}]")
        print(f"  Train rows: {len(X_tr)} (pits: {y_tr.sum()})")
        print(f"  Test rows:  {len(X_te)} (pits: {y_te.sum()})")
        print(f"  B1 Degradation Sensitivity : {round(clf.coef_[0][0], 4)}")
        print(f"  B2 Position Sensitivity    : {round(clf.coef_[0][1], 4)}")
        print(f"  B0 Intercept               : {round(clf.intercept_[0], 4)}")
        print(f"  Accuracy                   : {round(acc * 100, 2)}%")
        print(f"  ROC-AUC                    : {round(auc, 4) if not np.isnan(auc) else 'N/A'}")
        return clf, auc
 
    X_tr = train_beh[['DegradationPercent', 'Position']]
    y_tr = train_beh['Did_Pit']
    X_te = test_beh[['DegradationPercent', 'Position']]
    y_te = test_beh['Did_Pit']
 
    team_clf, team_auc = fit_and_evaluate(X_tr, y_tr, X_te, y_te,
                                          label=f"{target_team} TEAM LEVEL")
 
    # ── Stage D: Export models ────────────────────────────────────────────────
    name = target_team.replace(" ", "_").lower()
    scaler_path = os.path.join(EXPORT_DIR, f"{name}_scaler.joblib")
    gmm_path    = os.path.join(EXPORT_DIR, f"{name}_gmm.joblib")
    clf_path    = os.path.join(EXPORT_DIR, f"{name}_behavior.joblib")
 
    joblib.dump(scaler,   scaler_path)
    joblib.dump(gmm,      gmm_path)
    joblib.dump(team_clf, clf_path)
 
    print(f"\n  Exported:")
    print(f"    {scaler_path}")
    print(f"    {gmm_path}")
    print(f"    {clf_path}")
 
    # ── Stage E: Driver-Level Split (Bimodality Hypothesis Test) ─────────────
    if driver_level:
        print(f"\n  --- BIMODALITY HYPOTHESIS TEST: {target_team} ---")
 
        drivers = train_beh['DriverId'].unique()
        driver_aucs = {}
 
        for driver in drivers:
            d_tr = train_beh[train_beh['DriverId'] == driver]
            d_te = test_beh[test_beh['DriverId']   == driver]
 
            if (len(d_tr[d_tr['Did_Pit'] == 1]) < 3 or
                    len(d_te[d_te['Did_Pit'] == 1]) < 1):
                print(f"\n  [{driver}] insufficient data, skipping")
                continue
 
            _, d_auc = fit_and_evaluate(
                d_tr[['DegradationPercent', 'Position']], d_tr['Did_Pit'],
                d_te[['DegradationPercent', 'Position']], d_te['Did_Pit'],
                label=f"{driver} DRIVER LEVEL"
            )
            driver_aucs[driver] = d_auc
 
        if len(driver_aucs) == 2:
            drivers_list = list(driver_aucs.keys())
            auc_a = driver_aucs[drivers_list[0]]
            auc_b = driver_aucs[drivers_list[1]]
            delta = abs(auc_a - auc_b) if not (np.isnan(auc_a) or np.isnan(auc_b)) else np.nan
 
            print(f"\n  INTERPRETATION:")
            print(f"  Team AUC         : {round(team_auc, 4)}")
            print(f"  {drivers_list[0]} AUC : {round(auc_a, 4) if not np.isnan(auc_a) else 'N/A'}")
            print(f"  {drivers_list[1]} AUC : {round(auc_b, 4) if not np.isnan(auc_b) else 'N/A'}")
 
            if not np.isnan(delta):
                if delta > 0.10:
                    print(f"  >> BIMODALITY CONFIRMED (delta={round(delta,4)})")
                    print(f"     Low team AUC explained by contradictory driver strategies.")
                else:
                    print(f"  >> INFORMATION ASYMMETRY CONFIRMED (delta={round(delta,4)})")
                    print(f"     Both drivers individually unpredictable from public data.")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. MASTER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == '__main__':
    YEAR         = 2021
    TOTAL_ROUNDS = 22
    SPLIT_ROUND  = int(TOTAL_ROUNDS * 0.60)
 
    train_rounds = list(range(1, SPLIT_ROUND + 1))
    test_rounds  = list(range(SPLIT_ROUND + 1, TOTAL_ROUNDS + 1))
 
    print(f"Training on rounds  : {train_rounds}")
    print(f"Testing on rounds   : {test_rounds}")
 
    all_data = []
    for rnd in range(1, TOTAL_ROUNDS + 1):
        try:
            raw   = load_f1_data(YEAR, rnd)
            clean = filter_global_anomalies(raw)
            all_data.append(clean)
        except Exception as e:
            print(f"  Skipping round {rnd}: {e}")
 
    if not all_data:
        print("CRITICAL ERROR: No data loaded.")
        exit()
 
    full_df  = pd.concat(all_data, ignore_index=True)
    train_df = full_df[full_df['Round'].isin(train_rounds)]
    test_df  = full_df[full_df['Round'].isin(test_rounds)]
 
    print(f"\nFull season loaded: {len(full_df):,} laps")
    print(f"Train: {len(train_df):,} laps  |  Test: {len(test_df):,} laps")
 
    # Run all 10 teams
    all_teams = [
        'Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren',
        'Alpine', 'AlphaTauri', 'Aston Martin', 'Williams',
        'Alfa Romeo Racing', 'Haas F1 Team'
    ]
    for team in all_teams:
        process_team(train_df, test_df, team, driver_level=False)
 
    # Bimodality hypothesis test for top 2 teams
    print(f"\n{'#'*70}")
    print(f"# BIMODALITY HYPOTHESIS TEST")
    print(f"{'#'*70}")
    for team in ['Mercedes', 'Red Bull Racing']:
        process_team(train_df, test_df, team, driver_level=True)
 
    print(f"\n{'='*70}")
    print(f" PIPELINE COMPLETE")
    print(f" Models exported to: {EXPORT_DIR}")
    print(f"{'='*70}")
