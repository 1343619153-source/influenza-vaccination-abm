from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial, Poisson

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VAC_CSV = os.path.join(PROJECT_DIR, "data", "vaccination_2023_2025_by_group.csv")
ILI_CSV = os.path.join(PROJECT_DIR, "data", "weekly_ili_historical_inputs.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "negative_binomial")
PAPER_OUT_DIR = OUT_DIR
CI_ALPHA = 0.05

COUNT_MULTIPLIER = 500.0
MONTH_LABELS = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_EPI_WEEKS: Dict[int, List[int]] = {
    0: list(range(31, 35)),
    1: list(range(35, 39)),
    2: list(range(39, 44)),
    3: list(range(44, 48)),
    4: list(range(48, 53)),
    5: list(range(1, 5)),
    6: list(range(5, 9)),
    7: list(range(9, 14)),
}

TRAIN_YEARS = (23, 24)
TEST_YEAR = 25

NETWORK_PAPER_ERRORS = {
    "child": {"MAE": 1.05, "RMSE": 1.52, "sMAPE": 47.26},
    "adult": {"MAE": 0.96, "RMSE": 1.26, "sMAPE": 59.34},
    "elderly": {"MAE": 0.22, "RMSE": 0.26, "sMAPE": 27.69},
}

GROUPS = ("child", "adult", "elderly")
GROUP_LABELS = {"child": "Children", "adult": "Adults", "elderly": "Older Adults"}

_VAC_COL = {
    "child": {23: 1, 24: 2, 25: 3},
    "adult": {23: 4, 24: 5, 25: 6},
    "elderly": {23: 7, 24: 8, 25: 9},
}


def _read_csv_any(path: str) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to read CSV: {path} ({last_err})")


def _parse_pct(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    if isinstance(x, (int, float)):
        v = float(x)
        return v / 100.0 if v > 1.5 else v
    s = str(x).strip().replace("%", "")
    if not s:
        return float("nan")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v / 100.0 if v > 1.5 else v


def load_vaccination_scaled(path: str = VAC_CSV) -> Dict[int, Dict[str, np.ndarray]]:
    df = _read_csv_any(path)
    if df.shape[1] < 10 or df.shape[0] < 8:
        raise ValueError(
            f"Vaccination CSV requires at least 8 rows and 10 columns "
            f"(date plus three groups across three years); found {df.shape}"
        )
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for year in (23, 24, 25):
        out[year] = {}
        for group in GROUPS:
            col = _VAC_COL[group][year]
            arr = pd.to_numeric(df.iloc[:8, col], errors="coerce").to_numpy(dtype=float)
            if np.any(~np.isfinite(arr)):
                raise ValueError(
                    f"Vaccination CSV contains missing values: "
                    f"year={year}, group={group}, column={col}"
                )
            out[year][group] = arr
    return out


def load_ili_weekly(path: str = ILI_CSV) -> pd.DataFrame:
    df = _read_csv_any(path)
    if df.shape[1] < 9:
        raise ValueError(
            f"Weekly ILI input requires at least 9 columns; found {df.shape[1]}"
        )
    rows = []
    for _, row in df.iterrows():
        try:
            wk = int(float(row.iloc[0]))
        except Exception:
            continue
        rec = {"week": wk}
        for year, ili_i, pos_i in (
            (22, 1, 2),
            (23, 3, 4),
            (24, 5, 6),
            (25, 7, 8),
        ):
            rec[f"ili_{year}"] = _parse_pct(row.iloc[ili_i])
            rec[f"pos_{year}"] = _parse_pct(row.iloc[pos_i])
        rows.append(rec)
    out = pd.DataFrame(rows).dropna(subset=["week"])
    out["week"] = out["week"].astype(int)
    return out.sort_values("week").reset_index(drop=True)


def monthly_ili_features(ili_df: pd.DataFrame, year: int) -> Dict[str, np.ndarray]:
    ili_col = f"ili_{year}"
    pos_col = f"pos_{year}"
    prev_col = f"ili_{year - 1}"
    if ili_col not in ili_df.columns:
        raise KeyError(ili_col)

    ili_mean = np.zeros(8)
    pos_mean = np.zeros(8)
    for m, weeks in MONTH_EPI_WEEKS.items():
        sub = ili_df[ili_df["week"].isin(weeks)]
        ili_mean[m] = float(np.nanmean(sub[ili_col])) if len(sub) else 0.0
        pos_mean[m] = float(np.nanmean(sub[pos_col])) if len(sub) else 0.0

    ili_trend = np.zeros(8)
    for m in range(8):
        prev = ili_mean[m - 1] if m > 0 else ili_mean[0]
        ili_trend[m] = (ili_mean[m] - prev) / (prev + 1e-6)

    ili_rel = np.zeros(8)
    if prev_col in ili_df.columns:
        prev_mean = np.zeros(8)
        for m, weeks in MONTH_EPI_WEEKS.items():
            sub = ili_df[ili_df["week"].isin(weeks)]
            prev_mean[m] = float(np.nanmean(sub[prev_col])) if len(sub) else 0.0
        ili_rel = np.maximum(0.0, (ili_mean - prev_mean) / (prev_mean + 1e-6))

    return {
        "ili_mean": ili_mean,
        "pos_mean": pos_mean,
        "ili_trend": ili_trend,
        "ili_relative": ili_rel,
    }


@dataclass
class SeasonPanel:
    year: int
    month_idx: np.ndarray
    y: np.ndarray
    ili_mean: np.ndarray
    ili_trend: np.ndarray
    ili_relative: np.ndarray
    lag1: np.ndarray


def make_panels(
    series_by_year: Dict[int, Dict[str, np.ndarray]],
    ili_by_year: Dict[int, Dict[str, np.ndarray]],
    group: str,
) -> List[SeasonPanel]:
    panels: List[SeasonPanel] = []
    years_sorted = sorted(series_by_year.keys())
    for year in years_sorted:
        y = series_by_year[year][group].astype(float)
        ili = ili_by_year[year]
        lag = np.zeros(8)
        lag[0] = (
            float(series_by_year[year - 1][group][7])
            if (year - 1) in series_by_year
            else float(y[0])
        )
        lag[1:] = y[:-1]
        panels.append(
            SeasonPanel(
                year=year,
                month_idx=np.arange(8, dtype=int),
                y=y,
                ili_mean=ili["ili_mean"],
                ili_trend=ili["ili_trend"],
                ili_relative=ili["ili_relative"],
                lag1=lag,
            )
        )
    return panels


def stack_design(
    panels: Sequence[SeasonPanel],
    *,
    use_month_fe: bool,
    use_ili: bool,
    use_lag: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ys: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    for p in panels:
        n = len(p.y)
        cols: List[np.ndarray] = []
        names: List[str] = []
        if use_month_fe:
            for m in range(1, 8):
                cols.append((p.month_idx == m).astype(float))
                names.append(f"m{m}")
        if use_ili:
            cols.extend([p.ili_mean, p.ili_trend, p.ili_relative])
            names.extend(["ili_mean", "ili_trend", "ili_relative"])
        if use_lag:
            cols.append(p.lag1)
            names.append("lag1")
        if cols:
            x = np.column_stack(cols)
        else:
            x = np.zeros((n, 0))
        ys.append(p.y)
        xs.append(x)
        feature_names = names
    y = np.concatenate(ys)
    if xs[0].size == 0:
        X = np.zeros((len(y), 0))
    else:
        X = np.vstack(xs)
    X = sm.add_constant(X, has_constant="add")
    return y, X, ["const"] + feature_names


def _fit_count_glm(y_scaled: np.ndarray, X: np.ndarray) -> Tuple[Any, str]:
    y_scaled = np.asarray(y_scaled, dtype=float)
    y_scaled = np.clip(y_scaled, 0.0, None)
    y_int = np.rint(y_scaled * COUNT_MULTIPLIER).astype(int)
    try:
        model = sm.GLM(y_int, X, family=NegativeBinomial(alpha=1.0))
        res = model.fit(disp=0, maxiter=200)
        return res, "Negative Binomial"
    except Exception:
        model = sm.GLM(y_int, X, family=Poisson())
        res = model.fit(disp=0, maxiter=200)
        return res, "Poisson"


def predict_mean(res: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(res.predict(X), dtype=float)


def predict_mean_ci(
    res: Any, X: np.ndarray, alpha: float = CI_ALPHA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = res.get_prediction(X)
    sf = pred.summary_frame(alpha=alpha)
    mean = np.asarray(sf["mean"], dtype=float)
    lo = np.asarray(sf["mean_ci_lower"], dtype=float)
    hi = np.asarray(sf["mean_ci_upper"], dtype=float)
    lo = np.clip(lo, 0.0, None)
    return mean, lo, hi


@dataclass
class ModelSpec:
    key: str
    name: str
    use_month_fe: bool
    use_ili: bool
    use_lag: bool


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("M1_intercept", "M1 Intercept Only", False, False, False),
    ModelSpec("M2_monthFE", "M2 + Monthly Fixed Effects", True, False, False),
    ModelSpec("M3_monthFE_ILI", "M3 + Monthly FE + ILI", True, True, False),
    ModelSpec("M4_full", "M4 + Monthly FE + ILI + Lag", True, True, True),
]


def mae_rmse_smape(y_hat: np.ndarray, y_true: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    y_hat = np.asarray(y_hat, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    err = y_hat - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    smape = float(
        100.0
        * np.mean(2.0 * np.abs(err) / (np.abs(y_hat) + np.abs(y_true) + eps))
    )
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}


def run_group(
    group: str,
    train_panels: Sequence[SeasonPanel],
    test_panel: SeasonPanel,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y_true_scaled = np.asarray(test_panel.y, dtype=float)

    for spec in MODEL_SPECS:
        y_tr, X_tr, names = stack_design(
            train_panels,
            use_month_fe=spec.use_month_fe,
            use_ili=spec.use_ili,
            use_lag=spec.use_lag,
        )
        y_te, X_te, _ = stack_design(
            [test_panel],
            use_month_fe=spec.use_month_fe,
            use_ili=spec.use_ili,
            use_lag=spec.use_lag,
        )
        if X_te.shape[1] != X_tr.shape[1]:
            raise RuntimeError(f"{spec.key} design matrix column count mismatch")

        X_tr_fit = X_tr.copy()
        X_te_fit = X_te.copy()
        if "lag1" in names:
            j = names.index("lag1")
            X_tr_fit[:, j] = X_tr_fit[:, j] * COUNT_MULTIPLIER
            X_te_fit[:, j] = X_te_fit[:, j] * COUNT_MULTIPLIER

        res, fam = _fit_count_glm(y_tr, X_tr_fit)
        yhat_count, lo_count, hi_count = predict_mean_ci(res, X_te_fit, CI_ALPHA)
        yhat_scaled = yhat_count / COUNT_MULTIPLIER
        lo_scaled = lo_count / COUNT_MULTIPLIER
        hi_scaled = hi_count / COUNT_MULTIPLIER
        width_scaled = hi_scaled - lo_scaled
        metrics = mae_rmse_smape(yhat_scaled, y_true_scaled)

        rows.append(
            {
                "group": group,
                "group_label": GROUP_LABELS[group],
                "model": spec.key,
                "model_name": spec.name,
                "family": fam,
                "n_train": int(len(y_tr)),
                "AIC": float(getattr(res, "aic", np.nan)),
                "features": ",".join(names),
                **metrics,
                "yhat_scaled": yhat_scaled,
                "y_true_scaled": y_true_scaled,
                "ci_lower_scaled": lo_scaled,
                "ci_upper_scaled": hi_scaled,
                "ci_width_scaled": width_scaled,
                "mean_ci_width": float(np.mean(width_scaled)),
            }
        )
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Kunshan Influenza Vaccination: Nested Poisson / Negative Binomial Baselines")
    print("=" * 72)

    series = load_vaccination_scaled(VAC_CSV)
    ili_df = load_ili_weekly(ILI_CSV)
    ili_by_year = {
        y: monthly_ili_features(ili_df, y) for y in (22, 23, 24, 25)
    }

    print(f"\nVaccination CSV: {VAC_CSV}")
    print(f"ILI CSV:  {ILI_CSV}")
    print(
        f"Training seasons: {TRAIN_YEARS}  | Test season: {TEST_YEAR}  | "
        f"Values are scaled; count-model fitting multiplier: {COUNT_MULTIPLIER:g}"
    )
    print(
        "Column layout: Date | Children 2023/2024/2025 | "
        "Adults 2023/2024/2025 | Older Adults 2023/2024/2025"
    )

    print("\n[Scaled Observations for Test Year 2025]")
    for g in GROUPS:
        print(f"  {GROUP_LABELS[g]}: {np.round(series[TEST_YEAR][g], 3).tolist()}")

    all_rows: List[Dict[str, Any]] = []
    pred_long: List[Dict[str, Any]] = []

    for group in GROUPS:
        panels = make_panels(series, ili_by_year, group)
        train = [p for p in panels if p.year in TRAIN_YEARS]
        test = [p for p in panels if p.year == TEST_YEAR][0]
        if len(train) < 1:
            raise RuntimeError(f"{group}: no training panels")

        group_rows = run_group(group, train, test)
        all_rows.extend(group_rows)

        print(f"\n--- {GROUP_LABELS[group]} ---")
        for r in group_rows:
            print(
                f"  {r['model_name']:<32} [{r['family']}]  "
                f"MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  "
                f"sMAPE={r['sMAPE']:.2f}%  AIC={r['AIC']:.1f}"
            )
        net = NETWORK_PAPER_ERRORS[group]
        print(
            f"  {'Network Model (Published)':<32} [ABM]  "
            f"MAE={net['MAE']:.3f}  RMSE={net['RMSE']:.3f}  "
            f"sMAPE={net['sMAPE']:.2f}%"
        )

        for r in group_rows:
            for m, lab in enumerate(MONTH_LABELS):
                pred_long.append(
                    {
                        "group": group,
                        "group_label": r["group_label"],
                        "model": r["model"],
                        "model_name": r["model_name"],
                        "family": r["family"],
                        "month": lab,
                        "month_idx": m + 1,
                        "y_true_scaled": float(r["y_true_scaled"][m]),
                        "yhat_scaled": float(r["yhat_scaled"][m]),
                        "ci_lower_scaled": float(r["ci_lower_scaled"][m]),
                        "ci_upper_scaled": float(r["ci_upper_scaled"][m]),
                        "ci_width_scaled": float(r["ci_width_scaled"][m]),
                    }
                )

    summary = pd.DataFrame(
        [
            {
                "Group": r["group_label"],
                "Model": r["model_name"],
                "Family": r["family"],
                "MAE": round(r["MAE"], 4),
                "RMSE": round(r["RMSE"], 4),
                "sMAPE(%)": round(r["sMAPE"], 2),
                "AIC": round(r["AIC"], 2),
                "Training Sample Size": r["n_train"],
            }
            for r in all_rows
        ]
    )
    for g in GROUPS:
        net = NETWORK_PAPER_ERRORS[g]
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "Group": GROUP_LABELS[g],
                            "Model": "Network Model (Published)",
                            "Family": "ABM",
                            "MAE": net["MAE"],
                            "RMSE": net["RMSE"],
                            "sMAPE(%)": net["sMAPE"],
                            "AIC": np.nan,
                            "Training Sample Size": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary_path = os.path.join(OUT_DIR, "baseline_error_summary.csv")
    pred_path = os.path.join(OUT_DIR, "baseline_monthly_predictions.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pred_df = pd.DataFrame(pred_long)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    tex_lines = [
        "% Generated by negative_binomial_baseline.py using scaled values "
        "consistent with the network-model evaluation",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Model & Group & MAE & RMSE & sMAPE (\\%) \\\\",
        "\\midrule",
    ]
    for r in all_rows:
        tex_lines.append(
            f"{r['model_name']} & {r['group_label']} & "
            f"{r['MAE']:.2f} & {r['RMSE']:.2f} & {r['sMAPE']:.2f} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = os.path.join(OUT_DIR, "baseline_error_table_snippet.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    os.makedirs(PAPER_OUT_DIR, exist_ok=True)

    err_export = summary.copy()
    mean_w_map = {
        (r["group_label"], r["model_name"]): round(r["mean_ci_width"], 4)
        for r in all_rows
    }
    err_export["Mean CI Width"] = [
        mean_w_map.get((row["Group"], row["Model"]), np.nan)
        for _, row in err_export.iterrows()
    ]
    err_path = os.path.join(PAPER_OUT_DIR, "negative_binomial_group_errors.csv")
    err_export.to_csv(err_path, index=False, encoding="utf-8-sig")

    monthly_all = pred_df.rename(
        columns={
            "group_label": "Group",
            "model_name": "Model",
            "family": "Family",
            "month": "Month",
            "month_idx": "Month Index",
            "y_true_scaled": "Observed Scaled",
            "yhat_scaled": "Predicted Scaled",
            "ci_lower_scaled": "CI Lower Scaled",
            "ci_upper_scaled": "CI Upper Scaled",
            "ci_width_scaled": "CI Width Scaled",
        }
    )[
        [
            "Group",
            "Model",
            "Family",
            "Month",
            "Month Index",
            "Observed Scaled",
            "Predicted Scaled",
            "CI Lower Scaled",
            "CI Upper Scaled",
            "CI Width Scaled",
        ]
    ]
    monthly_all_path = os.path.join(
        PAPER_OUT_DIR, "negative_binomial_monthly_predictions_all_models.csv"
    )
    monthly_all.to_csv(monthly_all_path, index=False, encoding="utf-8-sig")

    monthly_m4 = monthly_all[
        monthly_all["Model"] == "M4 + Monthly FE + ILI + Lag"
    ].copy()
    monthly_m4_path = os.path.join(
        PAPER_OUT_DIR, "negative_binomial_m4_monthly_predictions.csv"
    )
    monthly_m4.to_csv(monthly_m4_path, index=False, encoding="utf-8-sig")

    err_m4 = err_export[
        err_export["Model"].isin(
            ["M4 + Monthly FE + ILI + Lag", "Network Model (Published)"]
        )
    ].copy()
    err_m4_path = os.path.join(
        PAPER_OUT_DIR, "negative_binomial_m4_and_network_group_errors.csv"
    )
    err_m4.to_csv(err_m4_path, index=False, encoding="utf-8-sig")

    if len(monthly_m4):
        pivot_lo = monthly_m4.pivot(
            index="Month Index", columns="Group", values="CI Lower Scaled"
        )
        pivot_hi = monthly_m4.pivot(
            index="Month Index", columns="Group", values="CI Upper Scaled"
        )
        pivot_w = monthly_m4.pivot(
            index="Month Index", columns="Group", values="CI Width Scaled"
        )
        pivot_hat = monthly_m4.pivot(
            index="Month Index", columns="Group", values="Predicted Scaled"
        )
        wide = pd.DataFrame({"Month": [MONTH_LABELS[i] for i in range(8)]})
        for group_label in ("Children", "Adults", "Older Adults"):
            if group_label in pivot_hat.columns:
                wide[f"{group_label} Predicted"] = pivot_hat[group_label].to_numpy()
                wide[f"{group_label} CI Lower"] = pivot_lo[group_label].to_numpy()
                wide[f"{group_label} CI Upper"] = pivot_hi[group_label].to_numpy()
                wide[f"{group_label} CI Width"] = pivot_w[group_label].to_numpy()
        wide_path = os.path.join(
            PAPER_OUT_DIR, "negative_binomial_m4_monthly_ci_wide.csv"
        )
        wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    else:
        wide_path = "(no M4 rows)"

    print("\n" + "=" * 72)
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"\nCore outputs:\n  {summary_path}\n  {pred_path}\n  {tex_path}")
    print(
        f"\nDetailed outputs:\n"
        f"  {err_path}\n"
        f"  {monthly_all_path}\n"
        f"  {monthly_m4_path}\n"
        f"  {err_m4_path}\n"
        f"  {wide_path}"
    )
    print(
        "\nInterpretation: if M1-M4 substantially underperform the network model "
        "for children, the result supports the value of the mechanistic model. "
        "If count baselines are already competitive for adults or older adults, "
        "emphasize scenario interpretability rather than uniformly higher accuracy."
    )


if __name__ == "__main__":
    main()
