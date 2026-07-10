from sklearn.ensemble import RandomForestRegressor


def build_baseline_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, random_state=42)
