import os
import fastf1
import joblib
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
 
# ─── SETUP ────────────────────────────────────────────────────────────────────
if os.path.exists('/kaggle/working'):
    CACHE_DIR  = '/kaggle/working/f1_cache'
    EXPORT_DIR = '/kaggle/working/driver_models'
else:
    CACHE_DIR  = './f1_cache'
    EXPORT_DIR = './driver_models'
 
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA EXTRACTION & FUEL NORMALIZATION
# (same as team pipeline)
# ══════════════════════════════════════════════════════════════════════════════
 
def load_f1_data(year: int, round_number: int) -> pd.DataFrame:
    print(f"--> Loading Round {round_number}...")
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
 
    df['IsPitLap'] = ~df['PitInTime'].isnull()
    df = df.sort_values(['DriverId', 'LapNumber'])
    df['Did_Pit'] = (
        df.groupby('DriverId')['IsPitLap']
          .shift(-1).fillna(False).astype(int)
    )
 
    total_laps = df['LapNumber'].max()
    df['FuelRemaining_kg'] = 110.0 * (1.0 - (df['LapNumber'] / total_laps))
    df['NormalizedPace_s'] = df['LapTime_s'] - (df['FuelRemaining_kg'] * 0.03)
 
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL ANOMALY FILTERING
# ══════════════════════════════════════════════════════════════════════════════
 
def filter_global_anomalies(df: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
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
# 3. DRIVER DNA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
 
def process_driver(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   driver_id: str,
                   team_name: str) -> dict:
    """
    Extract Decision DNA for one driver using their team's
    GMM physics engine — ensures degradation model is
    calibrated to the car, not the individual driver.
    """
 
    # ── Stage A: Team Physics Engine (GMM) ───────────────────────────────────
    # Use team-level data to train the physics model
    # so degradation baseline reflects car characteristics
    team_train = train_df[train_df['Team'] == team_name]
 
    train_racing = team_train[
        (team_train['IsAccurate'] == True) &
        (~team_train['IsPitLap'])
    ].dropna(subset=['NormalizedPace_s'])
 
    if len(train_racing) < 50:
        return None
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_racing[['TyreLife', 'NormalizedPace_s']])
 
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
 
    train_racing = train_racing.copy()
    train_racing['Regime'] = gmm.predict(X_scaled)
    regime_paces = train_racing.groupby('Regime')['NormalizedPace_s'].mean()
    cliff_regime = regime_paces.idxmax()
 
    # ── Stage B: Driver-Specific Behavioral Data ──────────────────────────────
    def prepare_driver_behavioral(df: pd.DataFrame, driver: str) -> pd.DataFrame:
        driver_df = df[df['DriverId'] == driver].copy()
        racing = driver_df[
            (driver_df['IsAccurate'] == True) &
            (~driver_df['IsPitLap'])
        ].dropna(subset=['NormalizedPace_s']).copy()
 
        if racing.empty:
            return pd.DataFrame()
 
        racing['Regime'] = gmm.predict(
            scaler.transform(racing[['TyreLife', 'NormalizedPace_s']])
        )
 
        baselines = []
        for (d, stint), group in racing.groupby(['DriverId', 'Stint']):
            non_cliff = group[group['Regime'] != cliff_regime]
            base = (non_cliff['NormalizedPace_s'].mean()
                    if not non_cliff.empty
                    else group['NormalizedPace_s'].mean())
            baselines.append({
                'DriverId':    d,
                'Stint':       stint,
                'BaselinePace': base
            })
 
        baseline_df = pd.DataFrame(baselines)
        if baseline_df.empty:
            return pd.DataFrame()
 
        merged = driver_df.merge(baseline_df, on=['DriverId', 'Stint'], how='left')
        merged = merged.dropna(subset=['BaselinePace', 'NormalizedPace_s'])
        merged['DegradationPercent'] = (
            (merged['NormalizedPace_s'] - merged['BaselinePace'])
            / merged['BaselinePace'] * 100
        )
 
        return merged[[
            'Round', 'DriverId', 'LapNumber',
            'DegradationPercent', 'Position', 'Did_Pit'
        ]].dropna()
 
    train_beh = prepare_driver_behavioral(train_df, driver_id)
    test_beh  = prepare_driver_behavioral(test_df,  driver_id)
 
    if (len(train_beh[train_beh['Did_Pit'] == 1]) < 3 or
            len(test_beh[test_beh['Did_Pit'] == 1]) < 1):
        return None
 
    # ── Stage C: Fit Driver Decision DNA ─────────────────────────────────────
    X_tr = train_beh[['DegradationPercent', 'Position']]
    y_tr = train_beh['Did_Pit']
    X_te = test_beh[['DegradationPercent', 'Position']]
    y_te = test_beh['Did_Pit']
 
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
 
    probs = clf.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, probs)
    except ValueError:
        auc = np.nan
 
    # ── Stage D: Export driver model ──────────────────────────────────────────
    clf_path = os.path.join(EXPORT_DIR, f"{driver_id.lower()}_behavior.joblib")
    joblib.dump(clf, clf_path)
 
    return {
        'Driver':    driver_id,
        'Team':      team_name,
        'B1_Degradation': round(clf.coef_[0][0], 4),
        'B2_Position':    round(clf.coef_[0][1], 4),
        'B0_Intercept':   round(clf.intercept_[0], 4),
        'AUC':            round(auc, 4) if not np.isnan(auc) else None,
        'Train_Pits':     int(y_tr.sum()),
        'Test_Pits':      int(y_te.sum()),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. MASTER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == '__main__':
    YEAR         = 2021
    TOTAL_ROUNDS = 22
    SPLIT_ROUND  = int(TOTAL_ROUNDS * 0.60)
 
    train_rounds = list(range(1, SPLIT_ROUND + 1))
    test_rounds  = list(range(SPLIT_ROUND + 1, TOTAL_ROUNDS + 1))
 
    # ── Load full season ──────────────────────────────────────────────────────
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
 
    # ── Get all driver-team pairs ─────────────────────────────────────────────
    driver_team_map = (
        full_df.groupby('DriverId')['Team']
               .agg(lambda x: x.value_counts().index[0])
               .reset_index()
               .values.tolist()
    )
 
    # ── Run driver pipeline ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" EXTRACTING DRIVER DNA — ALL 20 DRIVERS")
    print(f"{'='*70}")
 
    results = []
    for driver_id, team_name in driver_team_map:
        print(f"\n  Processing {driver_id} ({team_name})...")
        result = process_driver(train_df, test_df, driver_id, team_name)
        if result:
            results.append(result)
            print(f"  B1: {result['B1_Degradation']}  "
                  f"B2: {result['B2_Position']}  "
                  f"AUC: {result['AUC']}")
        else:
            print(f"  Skipped — insufficient data")
 
    # ── Results table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC', ascending=True)
 
    print(f"\n{'='*70}")
    print(f" DRIVER DECISION DNA — FULL GRID RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Driver':<6} {'Team':<20} {'B1':>8} {'B2':>8} {'AUC':>8}")
    print(f"{'-'*54}")
    for _, row in results_df.iterrows():
        print(f"{row['Driver']:<6} {row['Team']:<20} "
              f"{row['B1_Degradation']:>8.4f} "
              f"{row['B2_Position']:>8.4f} "
              f"{str(row['AUC']):>8}")
 
    # ── Save results to CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(EXPORT_DIR, 'driver_dna_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
 
    # ── Flag anomalies ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" ANOMALY DETECTION")
    print(f"{'='*70}")
 
    # Below random AUC
    below_random = results_df[results_df['AUC'] < 0.50]
    if not below_random.empty:
        print(f"\nDrivers with below-random AUC (< 0.50):")
        for _, row in below_random.iterrows():
            print(f"  {row['Driver']} ({row['Team']}) — AUC: {row['AUC']}")
 
    # Positive B2 (aggressive — stays out under pressure)
    positive_b2 = results_df[results_df['B2_Position'] > 0]
    if not positive_b2.empty:
        print(f"\nDrivers with positive B2 (aggressive — stays out under pressure):")
        for _, row in positive_b2.iterrows():
            print(f"  {row['Driver']} ({row['Team']}) — B2: {row['B2_Position']}")
 
    # Highest B2 absolute value (most position-reactive)
    most_reactive = results_df.nsmallest(3, 'B2_Position')
    print(f"\nMost position-reactive drivers (most negative B2):")
    for _, row in most_reactive.iterrows():
        print(f"  {row['Driver']} ({row['Team']}) — B2: {row['B2_Position']}")
 
    print(f"\n{'='*70}")
    print(f" PIPELINE COMPLETE")
    print(f" Models exported to: {EXPORT_DIR}")
    print(f"{'='*70}")
