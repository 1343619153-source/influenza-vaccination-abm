import os
import warnings
from itertools import product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'sarima_monthly_vaccination_inputs.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'sarima')
RANDOM_SEED = 42
EPSILON = 1e-8

n_simulations = 50
forecast_steps = 8
s = 12
target_months = [8, 9, 10, 11, 12, 1, 2, 3]
month_labels_plot = [f'Month {m}' for m in target_months]

POPULATION_CONFIGS = [
    {
        'name': 'Children',
        'column': 'child',
        'actual_points': [3, 15.6, 31.4, 27.9, 21.8, 2.4, 0.6, 0.8],
    },
    {
        'name': 'Adults',
        'column': 'adult',
        'actual_points': [0.5, 3.8, 7.4, 6, 4.3, 0.3, 0.07, 0.03],
    },
    {
        'name': 'Older Adults',
        'column': 'elderly',
        'actual_points': [0.05, 0.97, 2.2, 1.7, 0.87, 0.05, 0.02, 0.02],
    },
]


def grid_search_sarima(endog, p_range, d_range, q_range, P_range, D_range, Q_range, seasonal_period):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    results = []

    total = len(p_range) * len(d_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)
    print(f"\nStarting AIC grid search across {total} parameter combinations...")

    for idx, (p, d, q, P, D, Q) in enumerate(product(p_range, d_range, q_range, P_range, D_range, Q_range), 1):
        try:
            model = SARIMAX(
                endog=endog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            aic = fitted.aic
            results.append({
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, seasonal_period),
                'aic': aic,
                'bic': fitted.bic
            })
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, seasonal_period)
        except Exception:
            continue

        if idx % 20 == 0 or idx == total:
            print(f"  Search progress: {idx}/{total}")

    results.sort(key=lambda x: x['aic'])
    return best_order, best_seasonal_order, best_aic, results


def get_selected_indices(future_dates):
    selected_indices = []
    for target_m in target_months:
        for idx, forecast_date in enumerate(future_dates):
            if forecast_date.month == target_m and idx not in selected_indices:
                selected_indices.append(idx)
                break
    if len(selected_indices) < len(target_months):
        selected_indices = list(range(min(len(target_months), forecast_steps)))
    return selected_indices


def calc_errors(actual_points, forecast_points):
    mae = np.mean(np.abs(actual_points - forecast_points))
    rmse = np.sqrt(np.mean((actual_points - forecast_points) ** 2))
    smape = np.mean(2 * np.abs(forecast_points - actual_points) /
                    (np.abs(forecast_points) + np.abs(actual_points) + EPSILON)) * 100
    return mae, rmse, smape


def calc_interval_metrics(actual_points, ci_lower, ci_upper):
    in_interval = (ci_lower <= actual_points) & (actual_points <= ci_upper)
    coverage_rate = np.mean(in_interval) * 100
    avg_interval_width = np.mean(ci_upper - ci_lower)
    covered_count = int(np.sum(in_interval))
    total_count = len(actual_points)
    return coverage_rate, avg_interval_width, covered_count, total_count, in_interval


def run_population_forecast(config, endog, historical_dates):
    group_name = config['name']
    actual_points_plot = np.array(config['actual_points'], dtype=float)

    print("\n" + "=" * 70)
    print(f"[{group_name}] Historical data: {len(endog)} months")
    print("=" * 70)

    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)

    best_order, best_seasonal_order, best_aic, search_results = grid_search_sarima(
        endog, p_range, d_range, q_range, P_range, D_range, Q_range, s
    )
    if best_order is None:
        raise RuntimeError(f"No valid SARIMA parameter combination found for {group_name}")

    p, d, q = best_order
    P, D, Q, _ = best_seasonal_order

    print(f"\n[{group_name}] Grid search complete. Best parameters: SARIMA({p},{d},{q})({P},{D},{Q},{s})")
    print(f"Best AIC: {best_aic:.4f}")
    print("\nTop five parameter combinations by AIC:")
    for i, item in enumerate(search_results[:5], 1):
        po, so = item['order'], item['seasonal_order']
        print(f"  {i}. SARIMA{po}({so[0]},{so[1]},{so[2]},{so[3]})  AIC={item['aic']:.4f}  BIC={item['bic']:.4f}")

    model = SARIMAX(
        endog=endog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    print(f"\n[{group_name}] Training model...")
    fitted_model = model.fit(disp=False)
    print(f"Model training complete. AIC={fitted_model.aic:.4f}, BIC={fitted_model.bic:.4f}")

    print(f"\n[{group_name}] Forecasting the next {forecast_steps} months (August 2025 to March 2026)...")
    forecast_result = fitted_model.get_forecast(steps=forecast_steps)
    theoretical_forecast = forecast_result.predicted_mean
    if hasattr(theoretical_forecast, 'values'):
        theoretical_forecast = theoretical_forecast.values
    theoretical_forecast = np.asarray(theoretical_forecast)

    np.random.seed(RANDOM_SEED)
    print(f"[{group_name}] Running {n_simulations} forecast simulations...")
    simulated_forecasts = []
    noise_std = max(fitted_model.resid.std(), np.std(endog) * 0.1)

    for i in range(n_simulations):
        if (i + 1) % 10 == 0:
            print(f"  Simulation progress: {i + 1}/{n_simulations}")
        noise = np.random.normal(0, noise_std, forecast_steps)
        sim_forecast = theoretical_forecast + noise
        sim_forecast = np.maximum(sim_forecast, 0)
        simulated_forecasts.append(sim_forecast)

    simulated_forecasts = np.array(simulated_forecasts)
    forecast = np.mean(simulated_forecasts, axis=0)

    last_date = historical_dates[-1]
    future_dates = [last_date + relativedelta(months=i + 1) for i in range(forecast_steps)]

    ci_lower = np.maximum(np.percentile(simulated_forecasts, 2.5, axis=0), 0)
    ci_upper = np.percentile(simulated_forecasts, 97.5, axis=0)

    selected_indices = get_selected_indices(future_dates)
    plot_count = len(target_months)
    ci_lower_plot = ci_lower[selected_indices[:plot_count]]
    ci_upper_plot = ci_upper[selected_indices[:plot_count]]
    forecast_plot = forecast[selected_indices[:plot_count]]

    mae, rmse, smape = calc_errors(actual_points_plot, forecast_plot)
    coverage_rate, avg_interval_width, covered_count, total_count, in_interval = calc_interval_metrics(
        actual_points_plot, ci_lower_plot, ci_upper_plot)
    interval_width_plot = ci_upper_plot - ci_lower_plot

    print(f"\n[{group_name}] Monthly 95% confidence intervals "
          f"({n_simulations} simulations, 2.5th-97.5th percentiles):")
    for i in range(plot_count):
        covered_flag = 'Yes' if in_interval[i] else 'No'
        print(f"  {month_labels_plot[i]}: lower={ci_lower_plot[i]:.4f}, upper={ci_upper_plot[i]:.4f}, "
              f"interval width={interval_width_plot[i]:.4f}, actual={actual_points_plot[i]:.4f}, "
              f"simulation mean={forecast_plot[i]:.4f}, within interval={covered_flag}")

    print(f"\n[{group_name}] Forecast errors (August to March):")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  sMAPE: {smape:.2f}%")
    print(f"\n[{group_name}] Interval evaluation (August to March):")
    print(f"  Prediction interval coverage probability (PICP): "
          f"{coverage_rate:.1f}% ({covered_count}/{total_count})")
    print(f"  Mean prediction interval width (MPIW): {avg_interval_width:.4f}")

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(plot_count)
    bar_width = 0.55

    for i in range(plot_count):
        ax.bar(x[i], ci_upper_plot[i] - ci_lower_plot[i], bottom=ci_lower_plot[i],
               width=bar_width, color='#FFB366', alpha=0.75, edgecolor='none', zorder=1)

    for i in range(plot_count):
        val = actual_points_plot[i]
        in_ci = ci_lower_plot[i] <= val <= ci_upper_plot[i]
        color = 'green' if in_ci else 'red'
        ax.plot(x[i], val, marker='D', markersize=9, color=color, zorder=3)
        label = str(int(val)) if val == int(val) else f'{val:.2f}'.rstrip('0').rstrip('.')
        ax.text(x[i], val + max(actual_points_plot.max(), ci_upper_plot.max()) * 0.03, label,
                ha='center', va='bottom', fontsize=9, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels_plot)
    ax.set_xlabel('Month (August to March)', fontsize=12)
    ax.set_ylabel('New Vaccinations per Month', fontsize=12)
    ax.set_title(
        f'{group_name}: 95% Confidence Interval from {n_simulations} Simulations vs Actual Values\n'
        f'Best Model: SARIMA({p},{d},{q})({P},{D},{Q},{s}), AIC={fitted_model.aic:.2f}',
        fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    legend_handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=9,
                   label='Actual Value (Within Interval)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=9,
                   label='Actual Value (Outside Interval)'),
        mpatches.Patch(facecolor='#FFB366', alpha=0.75, label='Simulated 95% Confidence Interval'),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    ax.text(0.02, 0.98,
            f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nsMAPE = {smape:.2f}%\n'
            f'Coverage = {coverage_rate:.1f}%\nMean Interval Width = {avg_interval_width:.4f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    output_filename = os.path.join(
        OUTPUT_DIR, f'{group_name}_SARIMA({p},{d},{q})({P},{D},{Q},{s})_forecast.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[{group_name}] Figure saved to: {output_filename}")
    plt.show()
    plt.close()

    forecast_df = pd.DataFrame({
        'month': month_labels_plot,
        'actual_value': actual_points_plot,
        'simulated_forecast_mean': forecast_plot,
        'confidence_interval_lower_2.5_pct': ci_lower_plot,
        'confidence_interval_upper_97.5_pct': ci_upper_plot,
        'interval_width': interval_width_plot,
        'within_interval': ['Yes' if v else 'No' for v in in_interval],
    })
    forecast_df['MAE'] = mae
    forecast_df['RMSE'] = rmse
    forecast_df['sMAPE(%)'] = smape
    forecast_df['PICP(%)'] = coverage_rate
    forecast_df['MPIW'] = avg_interval_width
    full_forecast_df = pd.DataFrame({
        'date': [d.strftime('%Y/%m/%d') for d in future_dates],
        'month': [f'Month {d.month}' for d in future_dates],
        'simulated_forecast_mean': forecast,
        'theoretical_forecast': theoretical_forecast,
        'confidence_interval_lower_2.5_pct': ci_lower,
        'confidence_interval_upper_97.5_pct': ci_upper,
    })

    csv_filename = os.path.join(
        OUTPUT_DIR, f'{group_name}_SARIMA({p},{d},{q})({P},{D},{Q},{s})_forecast.csv')
    summary_filename = os.path.join(
        OUTPUT_DIR, f'{group_name}_SARIMA({p},{d},{q})({P},{D},{Q},{s})_forecast_summary.txt')

    forecast_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    full_forecast_df.to_csv(
        os.path.join(OUTPUT_DIR, f'{group_name}_SARIMA({p},{d},{q})({P},{D},{Q},{s})_all_forecast_steps.csv'),
        index=False, encoding='utf-8-sig')

    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f'Random seed: {RANDOM_SEED}\n')
        f.write(f'Population: {group_name}\n')
        f.write(f'Model: SARIMA({p},{d},{q})({P},{D},{Q},{s})\n')
        f.write(f'Number of simulations: {n_simulations}\n')
        f.write(f'AIC: {fitted_model.aic:.4f}\n\n')
        f.write('Monthly 95% confidence intervals:\n')
        for i in range(plot_count):
            covered_flag = 'Yes' if in_interval[i] else 'No'
            f.write(f'  {month_labels_plot[i]}: lower={ci_lower_plot[i]:.4f}, '
                    f'upper={ci_upper_plot[i]:.4f}, interval width={interval_width_plot[i]:.4f}, '
                    f'actual={actual_points_plot[i]:.4f}, simulation mean={forecast_plot[i]:.4f}, '
                    f'within interval={covered_flag}\n')
        f.write(f'\nMAE: {mae:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'sMAPE: {smape:.2f}%\n')
        f.write(f'Prediction interval coverage probability (PICP): '
                f'{coverage_rate:.1f}% ({covered_count}/{total_count})\n')
        f.write(f'Mean prediction interval width (MPIW): {avg_interval_width:.4f}\n')

    print(f"[{group_name}] Forecast results saved to: {csv_filename}")
    print(f"[{group_name}] Summary text saved to: {summary_filename}")

    return {
        'name': group_name,
        'order': (p, d, q, P, D, Q, s),
        'aic': fitted_model.aic,
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'coverage_rate': coverage_rate,
        'covered_count': covered_count,
        'total_count': total_count,
        'avg_interval_width': avg_interval_width,
    }


def main():
    import traceback

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    data['time'] = pd.to_datetime(data['time'])
    historical_dates = data['time'].dt.to_pydatetime().tolist()

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Historical data: {len(data)} months")
    print(f"Date range: {historical_dates[0].strftime('%Y-%m-%d')} "
          f"to {historical_dates[-1].strftime('%Y-%m-%d')}")
    print(f"Data columns: {list(data.columns)}")

    all_results = []
    try:
        for config in POPULATION_CONFIGS:
            if config['column'] not in data.columns:
                print(f"\nWarning: column {config['column']} is missing; skipping {config['name']}")
                continue
            endog = data[config['column']].values.astype(float)
            result = run_population_forecast(config, endog, historical_dates)
            all_results.append(result)
    except Exception as e:
        print(f"\nExecution failed: {e}")
        traceback.print_exc()

    if all_results:
        summary_rows = []
        for item in all_results:
            p, d, q, P, D, Q, s_val = item['order']
            summary_rows.append({
                'population': item['name'],
                'best_model': f'SARIMA({p},{d},{q})({P},{D},{Q},{s_val})',
                'AIC': item['aic'],
                'MAE': item['mae'],
                'RMSE': item['rmse'],
                'sMAPE(%)': item['smape'],
                'PICP(%)': item['coverage_rate'],
                'covered_months': f"{item['covered_count']}/{item['total_count']}",
                'MPIW': item['avg_interval_width'],
            })
        summary_df = pd.DataFrame(summary_rows)
        overall_summary = os.path.join(OUTPUT_DIR, 'population_groups_SARIMA_forecast_summary.csv')
        summary_df.to_csv(overall_summary, index=False, encoding='utf-8-sig')
        print(f"\nPopulation summary saved to: {overall_summary}")
        print("\nForecast error comparison across population groups:")
        print(summary_df.to_string(index=False))

    print("\nAll forecasts completed.")


if __name__ == '__main__':
    main()
