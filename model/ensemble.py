import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

from catboost import CatBoostRegressor

# optional models
HAS_LGBM = True
HAS_XGB = True

try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

df = pd.read_csv("Snowflake Notebooks Package/data/data_join.csv")

print("original shape:", df.shape)
print(df.head())



df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%m-%Y", errors="coerce")

# numeric columns
numeric_cols = [
    "LATITUDE", "LONGITUDE",
    "NIR", "GREEN", "SWIR16", "SWIR22",
    "NDMI", "MNDWI", "PET",
    "TOTALAL_KALINITY", "ELECTRICAL_CONDUCTANCE", "DISSOLVED_REACTIVE_PHOSPHORUS"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# sort by time
df = df.sort_values("DATE").reset_index(drop=True)

print("\nafter cleaning shape:", df.shape)
print("date range:", df["DATE"].min(), "->", df["DATE"].max())



def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    eps = 1e-6

    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["day"] = df["DATE"].dt.day
    df["dayofyear"] = df["DATE"].dt.dayofyear
    df["weekofyear"] = df["DATE"].dt.isocalendar().week.astype(float)
    df["quarter"] = df["DATE"].dt.quarter

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # -------------------------
    # spatial features
    # -------------------------
    df["lat_lon_interaction"] = df["LATITUDE"] * df["LONGITUDE"]
    df["lat_plus_lon"] = df["LATITUDE"] + df["LONGITUDE"]
    df["lat_minus_lon"] = df["LATITUDE"] - df["LONGITUDE"]
    df["lat_squared"] = df["LATITUDE"] ** 2
    df["lon_squared"] = df["LONGITUDE"] ** 2

    lat_mean = df["LATITUDE"].mean()
    lon_mean = df["LONGITUDE"].mean()

    df["lat_centered"] = df["LATITUDE"] - lat_mean
    df["lon_centered"] = df["LONGITUDE"] - lon_mean
    df["distance_from_center"] = np.sqrt(df["lat_centered"]**2 + df["lon_centered"]**2)


    band_pairs = [
        ("NIR", "GREEN"),
        ("NIR", "SWIR16"),
        ("NIR", "SWIR22"),
        ("GREEN", "SWIR16"),
        ("GREEN", "SWIR22"),
        ("SWIR16", "SWIR22"),
    ]

    for a, b in band_pairs:
        df[f"{a}_{b}_ratio"] = df[a] / (df[b] + eps)
        df[f"{a}_{b}_diff"] = df[a] - df[b]

    df["water_moisture_index"] = df["NDMI"] * df["MNDWI"]
    df["evaporation_stress_ndmi"] = df["PET"] / (np.abs(df["NDMI"]) + eps + 1.0)
    df["evaporation_stress_mndwi"] = df["PET"] / (np.abs(df["MNDWI"]) + eps + 1.0)

    df["pet_ndmi_product"] = df["PET"] * df["NDMI"]
    df["pet_mndwi_product"] = df["PET"] * df["MNDWI"]
    df["ndmi_mndwi_product"] = df["NDMI"] * df["MNDWI"]

    df["water_balance"] = df["MNDWI"] - df["NDMI"]
    df["is_water"] = (df["MNDWI"] > 0).astype(int)

    df["swir_sum"] = df["SWIR16"] + df["SWIR22"]
    df["nir_green_sum"] = df["NIR"] + df["GREEN"]

    df["nir_over_total"] = df["NIR"] / (df["NIR"] + df["GREEN"] + df["SWIR16"] + df["SWIR22"] + eps)
    df["green_over_total"] = df["GREEN"] / (df["NIR"] + df["GREEN"] + df["SWIR16"] + df["SWIR22"] + eps)
    df["swir_over_total"] = (df["SWIR16"] + df["SWIR22"]) / (df["NIR"] + df["GREEN"] + df["SWIR16"] + df["SWIR22"] + eps)

    df["lat_ndmi"] = df["LATITUDE"] * df["NDMI"]
    df["lon_ndmi"] = df["LONGITUDE"] * df["NDMI"]
    df["lat_pet"] = df["LATITUDE"] * df["PET"]
    df["lon_pet"] = df["LONGITUDE"] * df["PET"]

    return df


df = build_features(df)

targets = [
    "TOTALAL_KALINITY",
    "ELECTRICAL_CONDUCTANCE",
    "DISSOLVED_REACTIVE_PHOSPHORUS"
]

feature_cols = [
    c for c in df.columns
    if c not in ["DATE"] + targets
]

needed_cols = feature_cols + targets
df_model = df.dropna(subset=needed_cols).copy().reset_index(drop=True)

X = df_model[feature_cols].copy()

print("\nfinal model shape:", df_model.shape)
print("num features:", len(feature_cols))



def get_model_candidates(target_name: str):

    use_log = (target_name == "DISSOLVED_REACTIVE_PHOSPHORUS")

    candidates = []

 
    cat_grid = {
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05],
        "n_estimators": [500, 800, 1200],
        "l2_leaf_reg": [3, 5, 7, 9],
        "bagging_temperature": [0, 0.5, 1.0],
        "random_strength": [1, 2, 5],
    }
    candidates.append(("catboost", cat_grid))

    if HAS_LGBM:
        lgb_grid = {
            "n_estimators": [400, 800, 1200],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [-1, 6, 8, 10],
            "num_leaves": [31, 63],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0, 1, 3],
        }
        candidates.append(("lightgbm", lgb_grid))

  
    if HAS_XGB:
        xgb_grid = {
            "n_estimators": [400, 800],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1, 3, 5],
        }
        candidates.append(("xgboost", xgb_grid))


    hgb_grid = {
        "max_iter": [300, 600],
        "learning_rate": [0.03, 0.05],
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    candidates.append(("histgb", hgb_grid))

    return candidates, use_log


def build_model(model_name: str, params: dict):
    if model_name == "catboost":
        return CatBoostRegressor(
            loss_function="RMSE",
            random_state=42,
            verbose=False,
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            l2_leaf_reg=params["l2_leaf_reg"],
            bagging_temperature=params["bagging_temperature"],
            random_strength=params["random_strength"]
        )

    if model_name == "lightgbm":
        return LGBMRegressor(
            random_state=42,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"]
        )

    if model_name == "xgboost":
        return XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"]
        )

    if model_name == "histgb":
        return HistGradientBoostingRegressor(
            random_state=42,
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"]
        )

    raise ValueError(f"Unknown model name: {model_name}")


tscv = TimeSeriesSplit(n_splits=5)

def cross_validated_predictions(model_name, params, X, y, use_log=False):
    """
    Return:
        mean_r2
        fold_scores
        oof_predictions
    """
    oof_pred = np.zeros(len(y), dtype=float)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if use_log:
            y_train_fit = np.log1p(y_train)
        else:
            y_train_fit = y_train

        model = build_model(model_name, params)
        model.fit(X_train, y_train_fit)

        pred = model.predict(X_val)

        if use_log:
            pred = np.expm1(pred)

        pred = np.clip(pred, 0, None)

        oof_pred[val_idx] = pred
        fold_r2 = r2_score(y_val, pred)
        fold_scores.append(fold_r2)

    mean_r2 = np.mean(fold_scores)
    return mean_r2, fold_scores, oof_pred


def fit_full_model(model_name, params, X, y, use_log=False):
    if use_log:
        y_fit = np.log1p(y)
    else:
        y_fit = y

    model = build_model(model_name, params)
    model.fit(X, y_fit)
    return model


best_results = {}
final_models = {}
oof_predictions_per_target = {}

for target in targets:
    print(f"TARGET: {target}")

    y_target = df_model[target].copy().reset_index(drop=True)

    candidates, use_log = get_model_candidates(target)

    best_score = -np.inf
    best_model_name = None
    best_params = None
    best_oof = None
    best_fold_scores = None

    for model_name, grid in candidates:
        print(f"\nSearching model: {model_name}")

        for params in ParameterGrid(grid):
            try:
                mean_r2, fold_scores, oof_pred = cross_validated_predictions(
                    model_name=model_name,
                    params=params,
                    X=X,
                    y=y_target,
                    use_log=use_log
                )

                print(
                    f"{model_name} | params={params} | "
                    f"fold_r2={[round(s, 4) for s in fold_scores]} | "
                    f"mean_r2={mean_r2:.4f}"
                )

                if mean_r2 > best_score:
                    best_score = mean_r2
                    best_model_name = model_name
                    best_params = params
                    best_oof = oof_pred
                    best_fold_scores = fold_scores

            except Exception as e:
                print(f"skip {model_name} params={params} due to error: {e}")

    print("\nBEST RESULT")
    print("best model:", best_model_name)
    print("best params:", best_params)
    print("best CV R2:", round(best_score, 4))

    best_results[target] = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_cv_r2": best_score,
        "fold_scores": best_fold_scores,
        "use_log": use_log
    }

    oof_predictions_per_target[target] = best_oof

    final_models[target] = fit_full_model(
        model_name=best_model_name,
        params=best_params,
        X=X,
        y=y_target,
        use_log=use_log
    )



def get_top_models_for_target(target, top_k=3):
    y_target = df_model[target].copy().reset_index(drop=True)
    candidates, use_log = get_model_candidates(target)

    all_results = []

    for model_name, grid in candidates:
        for params in ParameterGrid(grid):
            try:
                mean_r2, fold_scores, oof_pred = cross_validated_predictions(
                    model_name=model_name,
                    params=params,
                    X=X,
                    y=y_target,
                    use_log=use_log
                )
                all_results.append({
                    "model_name": model_name,
                    "params": params,
                    "mean_r2": mean_r2,
                    "fold_scores": fold_scores,
                    "oof_pred": oof_pred,
                    "use_log": use_log
                })
            except Exception:
                pass

    all_results = sorted(all_results, key=lambda x: x["mean_r2"], reverse=True)
    return all_results[:top_k]


ensemble_results = {}
ensemble_final_models = {}

USE_ENSEMBLE = True

if USE_ENSEMBLE:
    for target in targets:
        print(f"ENSEMBLE SEARCH FOR TARGET: {target}")

        top_models = get_top_models_for_target(target, top_k=3)
        y_target = df_model[target].copy().reset_index(drop=True)

        if len(top_models) == 0:
            print("No models found for ensemble.")
            continue

        # simple average ensemble among top models
        all_oof = np.column_stack([m["oof_pred"] for m in top_models])
        ensemble_oof = all_oof.mean(axis=1)

        # valid OOF positions only
        valid_mask = np.any(all_oof != 0, axis=1)

        ensemble_r2 = r2_score(y_target[valid_mask], ensemble_oof[valid_mask])

        print("Top models:")
        for i, m in enumerate(top_models, start=1):
            print(
                f"{i}. {m['model_name']} | "
                f"mean_r2={m['mean_r2']:.4f} | params={m['params']}"
            )

        print(f"Simple ensemble OOF R2: {ensemble_r2:.4f}")

        # compare with single best
        single_best_r2 = best_results[target]["best_cv_r2"]

        if ensemble_r2 > single_best_r2:
            print("Ensemble is better. Using ensemble for final prediction.")

            fitted_models = []
            for m in top_models:
                fitted = fit_full_model(
                    model_name=m["model_name"],
                    params=m["params"],
                    X=X,
                    y=y_target,
                    use_log=m["use_log"]
                )
                fitted_models.append({
                    "model": fitted,
                    "model_name": m["model_name"],
                    "params": m["params"],
                    "use_log": m["use_log"]
                })

            ensemble_results[target] = {
                "type": "ensemble",
                "cv_r2": ensemble_r2,
                "members": top_models
            }
            ensemble_final_models[target] = fitted_models

        else:
            print("Single best model is better. Keeping single model.")
            ensemble_results[target] = {
                "type": "single",
                "cv_r2": single_best_r2
            }



print("FINAL SUMMARY")

final_scores = []

for target in targets:
    base_score = best_results[target]["best_cv_r2"]

    if USE_ENSEMBLE and target in ensemble_results and ensemble_results[target]["type"] == "ensemble":
        score_to_use = ensemble_results[target]["cv_r2"]
        mode = "ensemble"
    else:
        score_to_use = base_score
        mode = "single"

    final_scores.append(score_to_use)

    print(f"\nTarget: {target}")
    print(f"Mode: {mode}")
    print(f"Best Model: {best_results[target]['best_model_name']}")
    print(f"Best Params: {best_results[target]['best_params']}")
    print(f"Best CV R2: {best_results[target]['best_cv_r2']:.4f}")
    print(f"Fold Scores: {[round(x, 4) for x in best_results[target]['fold_scores']]}")

overall_mean_r2 = np.mean(final_scores)
print(f"\nOverall Average CV R2: {overall_mean_r2:.4f}")

def make_predictions(new_df: pd.DataFrame):
    """
    new_df must contain:
    DATE, LATITUDE, LONGITUDE, PET, NIR, GREEN, SWIR16, SWIR22, NDMI, MNDWI
    """
    temp = new_df.copy()

    temp["DATE"] = pd.to_datetime(temp["DATE"], format="%d-%m-%Y", errors="coerce")

    for col in [
        "LATITUDE", "LONGITUDE",
        "NIR", "GREEN", "SWIR16", "SWIR22",
        "NDMI", "MNDWI", "PET"
    ]:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    temp = build_features(temp)
    X_new = temp[feature_cols].copy()

    preds = pd.DataFrame(index=temp.index)

    for target in targets:
        # ensemble case
        if USE_ENSEMBLE and target in ensemble_final_models:
            pred_list = []

            for m in ensemble_final_models[target]:
                model = m["model"]
                use_log = m["use_log"]

                pred = model.predict(X_new)
                if use_log:
                    pred = np.expm1(pred)

                pred = np.clip(pred, 0, None)
                pred_list.append(pred)

            final_pred = np.mean(np.column_stack(pred_list), axis=1)
            preds[target] = final_pred

        else:
            model = final_models[target]
            use_log = best_results[target]["use_log"]

            pred = model.predict(X_new)
            if use_log:
                pred = np.expm1(pred)

            pred = np.clip(pred, 0, None)
            preds[target] = pred

    return preds


print("FEATURE IMPORTANCE (only if best model is CatBoost)")

for target in targets:
    if best_results[target]["best_model_name"] == "catboost":
        model = final_models[target]
        importances = pd.Series(model.get_feature_importance(), index=feature_cols)
        importances = importances.sort_values(ascending=False).head(20)

        print(f"\nTop 20 features for {target}")
        print(importances)