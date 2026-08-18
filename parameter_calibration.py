











from __future__ import annotations

import importlib.util
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from multiprocessing import current_process
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


_CPU_LOGICAL = max(1, os.cpu_count() or 16)




_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_PATH = os.path.join(_DIR, "historical_simulation.py")


def _load_main_module():
    spec = importlib.util.spec_from_file_location("kunshan_monthly_main", _MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {_MAIN_PATH}")
    km = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = km
    spec.loader.exec_module(km)
    return km


KM = _load_main_module()


_WORKER_NT: Optional[Dict[int, Any]] = None
_WORKER_MP: Optional[Dict[int, Dict[str, Any]]] = None
_MC_PROCESS_POOL: Optional[ProcessPoolExecutor] = None


_CALIB_LOCK = threading.Lock()


def _mp_init(
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    csv_path: str,
) -> None:
    global _WORKER_NT, _WORKER_MP
    load_calibration_ili_24_vs_23(KM, csv_path)
    _WORKER_NT = nodes_template
    _WORKER_MP = monthly_params


def _mp_task(payload: Tuple[int, Dict[str, Any]]) -> List[Dict[str, Any]]:

    seed, fp_dict = payload
    if _WORKER_NT is None or _WORKER_MP is None:
        raise RuntimeError("Worker process failed to initialize _WORKER_NT / _WORKER_MP")
    return run_single_simulation(KM, _WORKER_NT, _WORKER_MP, fp_dict, seed)


@contextmanager
def mc_pool_scope(
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
) -> Any:

    global _MC_PROCESS_POOL
    w = max(1, min(SIM_PARALLEL_WORKERS, N_SIM_RUNS_PER_EVAL))
    ex = ProcessPoolExecutor(
        max_workers=w,
        initializer=_mp_init,
        initargs=(nodes_template, monthly_params, CALIB_ILI_CSV),
    )
    _MC_PROCESS_POOL = ex
    try:
        yield ex
    finally:
        ex.shutdown(wait=True)
        _MC_PROCESS_POOL = None







REAL_MONTHLY_NEW_VACCINE = np.array(
    [
        [7.54, 0.895, 0.115],
        [27.0, 2.989, 0.807],
        [32.2, 4.408, 1.244],
        [23.96, 2.947, 0.879],
        [7.3, 1.348, 0.607],
        [4.22, 0.925, 0.191],
        [3.78, 0.675, 0.069],
        [1.15, 0.291, 0.075],
    ],
    dtype=float,
)


CALIB_ILI_CSV = os.path.join(_DIR, "data", "weekly_ili_historical_inputs.csv")


FIXED_P_SINGLE_INFECTION = 0.10


TARGET_PRIOR_RATIO_CHILD = 1.877
TARGET_PRIOR_RATIO_ELDERLY = 14.6
LAMBDA_PRIOR_PENALTY = 0.10


N_SIM_RUNS_PER_EVAL = 3
SIM_SEED_BASE = 424242


SIM_PARALLEL_WORKERS = max(1, min(N_SIM_RUNS_PER_EVAL, _CPU_LOGICAL))


PARAM_DIM = 19
PARAM_BOUNDS_LOW = np.array(
    [-5.0, -5.0, -5.0, 0.3, 0.3, 0.3]
    + [-7.0] * 8
    + [0.45, 2.0]
    + [0.2, 0.2, 0.2],
    dtype=float,
)
PARAM_BOUNDS_HIGH = np.array(
    [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
    + [0.0] * 8
    + [0.85, 3.3]
    + [0.8, 0.8, 0.8],
    dtype=float,
)

GA_POPSIZE = 80
GA_NGENERATIONS = 200
GA_SEED = 7
GA_CROSSOVER_RATE = 0.85
GA_MUTATION_RATE = 0.25
GA_MUTATION_SCALE = 0.15
GA_TOURNAMENT_SIZE = 3
GA_ELITE_COUNT = 5
GA_BLEND_ALPHA = 0.5

GA_POPULATION_PARALLEL = 1
GA_GENERATION_PRINT_INTERVAL = 5


PROGRESS_EVAL_INTERVAL = 0


def load_calibration_ili_24_vs_23(km: Any, csv_path: str) -> None:






    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Calibration ILI file not found: {csv_path}")

    df = km._read_weekly_ili_csv(csv_path)
    if df.shape[0] == 0 or df.shape[1] < 9:
        raise ValueError("Calibration requires the same nine-column wide-format CSV as historical_simulation.py")

    default_pos = float(km.ILI_CONFIG.get("default_weekly_positivity", 0.25))
    rows_ordered: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        try:
            wk = int(float(row.iloc[0]))
        except Exception:
            continue
        ili23 = km._parse_percent_value(row.iloc[3])
        ili24 = km._parse_percent_value(row.iloc[5])
        pos24 = km._parse_percent_value(row.iloc[6])
        if np.isnan(ili24) or np.isnan(ili23):
            continue
        if np.isnan(pos24):
            pos24 = default_pos

        rows_ordered.append(
            {
                "wk": wk,
                "ili23": float(ili23),
                "ili24": float(ili24),
                "pos24": float(pos24),
            }
        )

    if not rows_ordered:
        raise ValueError("Calibration ILI: no valid rows were parsed")

    lookup = {}
    auto_cache = {}
    for i, rec in enumerate(rows_ordered):
        wk = rec["wk"]
        p24 = rec["ili24"]
        h23 = rec["ili23"]
        pos24 = rec["pos24"]

        if i > 0:
            p_prev = rows_ordered[i - 1]["ili24"]
            ili_trend = (p24 - p_prev) / (p_prev + 1e-6)
        else:
            ili_trend = 0.0

        ili_relative = max(0.0, (p24 - h23) / (h23 + 1e-6))

        lookup[wk] = {"ili_trend": float(ili_trend), "ili_relative": float(ili_relative)}
        auto_cache[wk] = {"ili_tot": p24, "p_pos": float(pos24)}

    km.VACCINATION_ILI_WEEKLY_LOOKUP_CACHE = lookup
    km.ILI_AUTONOMOUS_WEEK_CACHE = auto_cache
    km.ILI_WEEKLY_CSV_LOAD_FAILED = False
    km.update_vaccination_feature_stats_from_ili_lookup(lookup)
    km._fill_monthly_infection_autonomous_from_cache()


def make_monthly_params_fixed_pn(km: Any, p_n: float) -> Dict[int, Dict[str, Any]]:

    out = {}
    for m in range(1, km.SIMULATION_MONTHS + 1):
        d = dict(km.MONTHLY_PARAMS[m])
        d["p_single_infection"] = float(p_n)
        out[m] = d
    return out


def vector_to_formula_dict(vec: np.ndarray) -> Dict[str, Any]:

    vec = np.asarray(vec, dtype=float).ravel()
    if vec.size != PARAM_DIM:
        raise ValueError(f"Parameter vector length must be {PARAM_DIM}; received {vec.size}")
    return {
        "b_child": float(vec[0]),
        "b_adult": float(vec[1]),
        "b_elderly": float(vec[2]),
        "s_child": float(vec[3]),
        "s_adult": float(vec[4]),
        "s_elderly": float(vec[5]),
        "policy_effects": [float(x) for x in vec[6:14]],
        "prior_child": float(vec[14]),
        "prior_adult": 0.0,
        "prior_elderly": float(vec[15]),
        "phi_child": float(vec[16]),
        "phi_adult": float(vec[17]),
        "phi_elderly": float(vec[18]),
    }


def formula_dict_to_vector(fp: Dict[str, Any]) -> np.ndarray:
    pe = fp.get("policy_effects") or [0.0] * 8
    if len(pe) < 8:
        pe = list(pe) + [0.0] * (8 - len(pe))
    return np.array(
        [
            fp.get("b_child", -2.0),
            fp.get("b_adult", -4.0),
            fp.get("b_elderly", -4.5),
            fp.get("s_child", 1.0),
            fp.get("s_adult", 1.0),
            fp.get("s_elderly", 1.0),
            *[float(x) for x in pe[:8]],
            fp.get("prior_child", 0.63),
            fp.get("prior_elderly", 2.68),
            fp.get("phi_child", 0.5),
            fp.get("phi_adult", 0.5),
            fp.get("phi_elderly", 0.5),
        ],
        dtype=float,
    )


def prior_penalty(fp: Dict[str, Any]) -> float:
    r_c = KM.representative_prior_vaccination_ratio(fp, "child")
    r_e = KM.representative_prior_vaccination_ratio(fp, "elderly")
    return float(
        (r_c - TARGET_PRIOR_RATIO_CHILD) ** 2
        + (r_e - TARGET_PRIOR_RATIO_ELDERLY) ** 2
    )


def build_nodes_template(km: Any, seed: int = 42) -> Dict[int, Any]:

    rng = random.Random(seed)
    family_dist = km.calculate_family_distribution()
    nodes, child_indices, _, _, _ = km.create_node_network(family_dist, rng=rng)
    nodes = km.add_school_connections(nodes, child_indices, rng=rng)
    nodes = km.add_company_connections(nodes, rng=rng)
    nodes = km.add_elderly_connections(nodes, rng=rng)

    template = {}
    for node_id, node_info in nodes.items():
        template[node_id] = node_info.copy()
        template[node_id]["health_status"] = None
    return template


def run_single_simulation(
    km: Any,
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    formula_params: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    km.VACCINATION_FORMULA_PARAMS_CACHE = formula_params
    rng = random.Random(seed)
    nodes_copy = {}
    for node_id, node_info in nodes_template.items():
        nodes_copy[node_id] = node_info.copy()
        nodes_copy[node_id]["health_status"] = None
    nodes_copy, _ = km.assign_health_status(nodes_copy, rng=rng)
    km.assign_prior_vaccination_history(nodes_copy, rng=rng)
    km.assign_cognition_traits(nodes_copy, rng=rng)
    return km.run_epidemic_simulation(
        nodes_copy,
        num_steps=km.DEFAULT_SIMULATION_STEPS,
        display_interval=10**9,
        verbose=False,
        rng=rng,
        monthly_params=monthly_params,
    )


def monthly_new_vaccine_matrix(
    km: Any, all_ts: Sequence[List[Dict[str, Any]]]
) -> np.ndarray:

    inc = km.calculate_monthly_increments(list(all_ts), num_runs=len(all_ts))
    mat = np.zeros((km.SIMULATION_MONTHS, 3), dtype=float)
    for m in range(1, km.SIMULATION_MONTHS + 1):
        if m not in inc:
            continue
        row = inc[m]
        mat[m - 1, 0] = row["child_vaccinated_new"]
        mat[m - 1, 1] = row["adult_vaccinated_new"]
        mat[m - 1, 2] = row["elderly_vaccinated_new"]
    return mat


def loss_rmse(pred: np.ndarray, real: np.ndarray, eps: float = 1e-9) -> float:

    scale = np.maximum(np.abs(real), eps)
    diff = (pred - real) / scale
    return float(np.sqrt(np.mean(diff ** 2)))


def _one_mc_run(
    km: Any,
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    fp: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    return run_single_simulation(km, nodes_template, monthly_params, fp, seed)



_OPT_STATE = {
    "obj_calls": 0,
    "t_opt_start": 0.0,
    "best_f": float("inf"),
    "best_x": None,
}


def _print_ga_iteration_progress(iteration: int) -> None:

    global_best = float(_OPT_STATE.get("best_f", float("inf")))
    print(
        f"  Generation {iteration}/{GA_NGENERATIONS}, best loss so far={global_best:.6f}",
        flush=True,
    )
    if iteration % GA_GENERATION_PRINT_INTERVAL != 0:
        return
    bx = _OPT_STATE.get("best_x")
    nfev = _OPT_STATE.get("obj_calls", 0)
    if bx is not None and np.isfinite(global_best):
        print(_format_snapshot_best(np.asarray(bx, dtype=float), global_best, nfev), flush=True)


def _format_snapshot_best(vec: np.ndarray, loss: float, eval_idx: int) -> str:

    fp = vector_to_formula_dict(vec)
    vec_s = np.array2string(
        np.asarray(vec, dtype=float),
        precision=8,
        separator=", ",
        max_line_width=120,
    )
    r_c = KM.representative_prior_vaccination_ratio(fp, "child")
    r_e = KM.representative_prior_vaccination_ratio(fp, "elderly")
    lines = [
        "",
        "─" * 72,
        f"[Objective evaluation #{eval_idx}] Best loss so far={loss:.6f}",
        f"  19-dimensional parameter vector: {vec_s}",
        f"  Prior multiplier (median features): child={r_c:.3f} (target≈{TARGET_PRIOR_RATIO_CHILD}), "
        f"elderly={r_e:.3f} (target≈{TARGET_PRIOR_RATIO_ELDERLY})",
        "  FORMULA_PARAMS dictionary:",
        f'    "b_child": {fp["b_child"]},',
        f'    "b_adult": {fp["b_adult"]},',
        f'    "b_elderly": {fp["b_elderly"]},',
        f'    "s_child": {fp["s_child"]},',
        f'    "s_adult": {fp["s_adult"]},',
        f'    "s_elderly": {fp["s_elderly"]},',
        f'    "phi_child": {fp["phi_child"]},',
        f'    "phi_adult": {fp["phi_adult"]},',
        f'    "phi_elderly": {fp["phi_elderly"]},',
        f'    "prior_child": {fp["prior_child"]},',
        f'    "prior_adult": 0.0,',
        f'    "prior_elderly": {fp["prior_elderly"]},',
        '    "policy_effects": [',
    ]
    pe = fp["policy_effects"]
    for i, v in enumerate(pe):
        comma = "," if i < len(pe) - 1 else ""
        lines.append(f"        {v}{comma}")
    lines.extend(
        [
            "    ],",
            f"  (P_n is fixed at {FIXED_P_SINGLE_INFECTION}; remember to set it in MONTHLY_PARAMS.)",
            "─" * 72,
            "",
        ]
    )
    return "\n".join(lines)


def objective_vector(
    vec: np.ndarray,
    km: Any,
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    target: np.ndarray,
) -> float:
    fp = vector_to_formula_dict(vec)
    seeds = [
        SIM_SEED_BASE + r * 9973 + int(abs(vec[0]) * 1000) % 100
        for r in range(N_SIM_RUNS_PER_EVAL)
    ]
    workers = max(1, min(SIM_PARALLEL_WORKERS, len(seeds)))
    if _MC_PROCESS_POOL is not None:
        payloads = [(s, fp) for s in seeds]
        runs = list(_MC_PROCESS_POOL.map(_mp_task, payloads))
    else:
        runner = partial(_one_mc_run, km, nodes_template, monthly_params, fp)
        if workers <= 1:
            runs = [runner(s) for s in seeds]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                runs = list(ex.map(runner, seeds))
    pred = monthly_new_vaccine_matrix(km, runs)
    loss = loss_rmse(pred, target) + LAMBDA_PRIOR_PENALTY * prior_penalty(fp)

    with _CALIB_LOCK:
        _OPT_STATE["obj_calls"] += 1
        if loss < _OPT_STATE["best_f"]:
            _OPT_STATE["best_f"] = float(loss)
            _OPT_STATE["best_x"] = np.asarray(vec, dtype=float).copy()

    return loss


def _clip_to_bounds(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), PARAM_BOUNDS_LOW, PARAM_BOUNDS_HIGH)


def _random_individual(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(PARAM_BOUNDS_LOW, PARAM_BOUNDS_HIGH)


def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    k: int,
) -> np.ndarray:
    idx = rng.integers(0, len(population), size=k)
    best_i = int(idx[np.argmin(fitness[idx])])
    return population[best_i].copy()


def _crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() >= GA_CROSSOVER_RATE:
        return parent1.copy() if rng.random() < 0.5 else parent2.copy()
    alpha = rng.uniform(-GA_BLEND_ALPHA, 1.0 + GA_BLEND_ALPHA, size=PARAM_DIM)
    child = alpha * parent1 + (1.0 - alpha) * parent2
    return _clip_to_bounds(child)


def _mutate(individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = individual.copy()
    span = PARAM_BOUNDS_HIGH - PARAM_BOUNDS_LOW
    for j in range(PARAM_DIM):
        if rng.random() < GA_MUTATION_RATE:
            delta = rng.uniform(-GA_MUTATION_SCALE, GA_MUTATION_SCALE) * span[j]
            x[j] += delta
    return _clip_to_bounds(x)


def _evaluate_population(
    population: np.ndarray,
    km: Any,
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    target: np.ndarray,
) -> np.ndarray:

    n = len(population)
    fitness = np.empty(n, dtype=float)
    for i in range(n):
        fitness[i] = float(
            objective_vector(population[i], km, nodes_template, monthly_params, target)
        )
    return fitness


def run_genetic_algorithm(
    km: Any,
    nodes_template: Dict[int, Any],
    monthly_params: Dict[int, Dict[str, Any]],
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(GA_SEED)

    _OPT_STATE["obj_calls"] = 0
    _OPT_STATE["best_f"] = float("inf")
    _OPT_STATE["best_x"] = None
    _OPT_STATE["t_opt_start"] = time.perf_counter()

    per_gen_evals = GA_POPSIZE
    total_evals_est = per_gen_evals * GA_NGENERATIONS
    total_sims_est = total_evals_est * N_SIM_RUNS_PER_EVAL
    print(
        f"\n>>> Genetic algorithm started: dimensions={PARAM_DIM}, "
        f"population={GA_POPSIZE}, maximum generations={GA_NGENERATIONS}, "
        f"{N_SIM_RUNS_PER_EVAL} MC runs per evaluation, parallel MC processes={SIM_PARALLEL_WORKERS}",
        flush=True,
    )
    print(
        f"    Estimated objective evaluations ≈ {total_evals_est}; full simulations ≈ {total_sims_est}"
        f" (480 steps per simulation); each generation reports generation x/{GA_NGENERATIONS} and the best loss so far; "
        f"full best parameters are also reported every {GA_GENERATION_PRINT_INTERVAL} generations.\n",
        flush=True,
    )

    population = np.array([_random_individual(rng) for _ in range(GA_POPSIZE)])
    seed_vec = _clip_to_bounds(formula_dict_to_vector(km.FORMULA_PARAMS))
    population[0] = seed_vec
    best_x: Optional[np.ndarray] = None
    best_f = float("inf")

    with mc_pool_scope(nodes_template, monthly_params):
        fitness = _evaluate_population(
            population,
            km,
            nodes_template,
            monthly_params,
            target,
        )
        for iteration in range(1, GA_NGENERATIONS + 1):
            order = np.argsort(fitness)
            gen_best_idx = int(order[0])
            gen_best_f = float(fitness[gen_best_idx])
            if gen_best_f < best_f:
                best_f = gen_best_f
                best_x = population[gen_best_idx].copy()

            _print_ga_iteration_progress(iteration)

            if iteration >= GA_NGENERATIONS:
                break

            next_pop: List[np.ndarray] = []
            for i in range(GA_ELITE_COUNT):
                next_pop.append(population[int(order[i])].copy())

            while len(next_pop) < GA_POPSIZE:
                p1 = _tournament_select(population, fitness, rng, GA_TOURNAMENT_SIZE)
                p2 = _tournament_select(population, fitness, rng, GA_TOURNAMENT_SIZE)
                child = _crossover(p1, p2, rng)
                child = _mutate(child, rng)
                next_pop.append(child)

            population = np.array(next_pop[:GA_POPSIZE])
            fitness = _evaluate_population(
                population,
                km,
                nodes_template,
                monthly_params,
                target,
            )

    if best_x is None:
        best_x = population[int(np.argmin(fitness))].copy()
        best_f = float(fitness[np.argmin(fitness)])

    bx = _OPT_STATE.get("best_x")
    bf = _OPT_STATE.get("best_f", float("inf"))
    if bx is not None and np.isfinite(bf) and bf < best_f:
        best_x = np.asarray(bx, dtype=float).copy()
        best_f = float(bf)

    elapsed = time.perf_counter() - _OPT_STATE["t_opt_start"]
    print(flush=True)
    print(
        f"\n>>> Genetic algorithm completed: best loss={best_f:.6f}, "
        f"objective evaluations={_OPT_STATE['obj_calls']}, total elapsed time={elapsed:.1f}s\n",
        flush=True,
    )
    return best_x, best_f


def print_results_for_manual_copy(km: Any, vec: np.ndarray, final_loss: float) -> None:
    fp = vector_to_formula_dict(vec)
    r_c = km.representative_prior_vaccination_ratio(fp, "child")
    r_e = km.representative_prior_vaccination_ratio(fp, "elderly")
    print("\n" + "=" * 72)
    print(f"Calibration completed: weighted RMSE + prior penalty loss = {final_loss:.6f}")
    print(f"Survey weights (fixed; see SURVEY_WEIGHT_CONFIG in historical_simulation.py):")
    for grp, label in [("child", "Child"), ("adult", "Adult"), ("elderly", "Elderly")]:
        w = km.SURVEY_WEIGHT_CONFIG.get(grp, {})
        print(
            f"  {label}: flu={w.get('flu', 0):.3f}, peer={w.get('peer', 0):.3f}, "
            f"pol={w.get('pol', 0):.3f}, cog={w.get('cog', 0):.3f}"
        )
    print(f"Prior multipliers: Child={r_c:.3f} (target≈{TARGET_PRIOR_RATIO_CHILD}), "
          f"elderly={r_e:.3f} (target≈{TARGET_PRIOR_RATIO_ELDERLY})")
    print("Copy the following dictionary into FORMULA_PARAMS = {...} in historical_simulation.py")
    print("-" * 72)
    print("FORMULA_PARAMS = {")
    print(f'    "b_child": {fp["b_child"]},')
    print(f'    "b_adult": {fp["b_adult"]},')
    print(f'    "b_elderly": {fp["b_elderly"]},')
    print(f'    "s_child": {fp["s_child"]},')
    print(f'    "s_adult": {fp["s_adult"]},')
    print(f'    "s_elderly": {fp["s_elderly"]},')
    print(f'    "phi_child": {fp["phi_child"]},')
    print(f'    "phi_adult": {fp["phi_adult"]},')
    print(f'    "phi_elderly": {fp["phi_elderly"]},')
    print(f'    "prior_child": {fp["prior_child"]},')
    print(f'    "prior_adult": 0.0,')
    print(f'    "prior_elderly": {fp["prior_elderly"]},')
    print('    "policy_effects": [')
    for i, v in enumerate(fp["policy_effects"]):
        comma = "," if i < len(fp["policy_effects"]) - 1 else ""
        print(f"        {v}{comma}  # Month {i + 1}")
    print("    ],")
    print("}")
    print("-" * 72)
    print(f"Set p_single_infection to {FIXED_P_SINGLE_INFECTION} for every month (fixed at 10% for this calibration)")
    print("=" * 72)


def validate_target(target: np.ndarray) -> None:
    if target.shape != (8, 3):
        raise ValueError(
            f"REAL_MONTHLY_NEW_VACCINE must have shape (8, 3) with columns [child, adult, elderly]; received {target.shape}"
        )
    if not np.any(target > 0):
        print("Warning: REAL_MONTHLY_NEW_VACCINE contains only zeros; enter the actual monthly new vaccinations for 2024.")


def main() -> None:
    print("\n" + "=" * 72)
    print("Parameter Calibration")
    print("=" * 72)

    print("\n[1/5] Validating the observed vaccination matrix REAL_MONTHLY_NEW_VACCINE...")
    validate_target(REAL_MONTHLY_NEW_VACCINE)

    print(
        f"\n[2/5] Parallel execution: logical CPUs ≈ {_CPU_LOGICAL}; "
        f"each objective evaluation uses {SIM_PARALLEL_WORKERS} processes × {N_SIM_RUNS_PER_EVAL} Monte Carlo runs (process pool)"
    )

    print("\n[3/5] Loading calibration ILI data (2024 vs 2023)...")
    load_calibration_ili_24_vs_23(KM, CALIB_ILI_CSV)
    print("      ILI caches loaded into historical_simulation.py.")

    monthly_params = make_monthly_params_fixed_pn(KM, FIXED_P_SINGLE_INFECTION)
    print(f"\n      P_n fixed at {FIXED_P_SINGLE_INFECTION}")

    print("\n[4/5] Building the network template used by historical_simulation.py...")
    nodes_template = build_nodes_template(KM, seed=KM.SIMULATION_CONFIG.get("network_seed", 42))
    print("      Network template ready.")

    print("\n[5/5] Running numerical optimization (genetic algorithm)...")
    x_best, f_best = run_genetic_algorithm(
        KM, nodes_template, monthly_params, REAL_MONTHLY_NEW_VACCINE
    )

    print_results_for_manual_copy(KM, x_best, f_best)


    print("\nAdditional validation: repeating simulations in parallel with the best parameters and aggregating the prediction matrix...")
    KM.VACCINATION_FORMULA_PARAMS_CACHE = vector_to_formula_dict(x_best)
    seeds_verify = [SIM_SEED_BASE + i * 10007 for i in range(N_SIM_RUNS_PER_EVAL)]
    fp_best = vector_to_formula_dict(x_best)
    payloads_v = [(s, fp_best) for s in seeds_verify]
    with mc_pool_scope(nodes_template, monthly_params) as pool:
        verify_runs = list(pool.map(_mp_task, payloads_v))
    pred_mat = monthly_new_vaccine_matrix(KM, verify_runs)
    print("\nPrediction mean across simulations: monthly new vaccinations for 8 months × [child, adult, elderly]:")
    print(pred_mat)
    print("\nObserved target (8 × 3):")
    print(REAL_MONTHLY_NEW_VACCINE)


if __name__ == "__main__" and current_process().name == "MainProcess":
    main()
