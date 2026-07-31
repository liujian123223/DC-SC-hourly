
import argparse
import multiprocessing
import datetime
import importlib.util
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import make_dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_components.task import TASK_TYPE_DEFAULTS
from utils.config_loader import load_yaml
from utils.transmission_cost_loader import load_transmission_matrix
from utils.transmission_region_mapper import map_location_to_region


OPTION_FIELD_DTYPES = {"option_task": np.int32, "option_start": np.int16, "option_finish": np.int16, "option_dest": np.int16, "option_tx_delay_hours": np.int16, "option_pue": float, "option_compute_energy_kwh": float, "option_facility_energy_kwh": float, "option_compute_carbon_kg": float, "option_direct_water_m3": float, "option_grid_water_m3": float, "option_water_m3": float, "option_transmission_energy_kwh": float, "option_transmission_carbon_kg": float, "option_carbon_kg": float, "option_tx_cost_usd": float}

OptionData = make_dataclass("OptionData", [(field, np.ndarray) for field in OPTION_FIELD_DTYPES] + [("option_objective", np.ndarray), ("task_option_groups", list), ("task_id_to_group", dict)], namespace={"__module__": __name__})

_OPTION_WORKER_CONTEXT = None

ASSIGNMENT_COLUMNS = ["task_id", "task_type", "arrival_hour", "start_hour", "finish_hour", "delay_hours", "origin", "location", "duration_minutes", "run_hours", "cores_req", "gpu_req", "gpu_mem_req", "mem_req", "bandwidth_gb", "source_task_count", "fraction", "pue", "compute_energy_kwh_proxy", "facility_energy_kwh_pue", "compute_carbon_kg", "direct_water_m3", "grid_water_m3", "water_m3", "transmission_energy_kwh", "transmission_carbon_kg", "carbon_kg", "tx_cost", "objective"]

SUMMARY_COLUMNS = [
    "strategy",
    "runtime_seconds",
    "objective",
    "compute_energy_kwh_proxy",
    "facility_energy_kwh_pue",
    "compute_carbon_kg",
    "tx_cost_usd",
    "load_weighted_delay_hours",
    "max_delay_hours",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "mem_energy_kwh",
    "training_compute_energy_kwh_proxy",
    "training_facility_energy_kwh_pue",
    "training_compute_carbon_kg",
    "training_mean_compute_power_kw",
    "training_peak_compute_power_kw",
    "training_mean_facility_power_kw",
    "training_peak_facility_power_kw",
    "inference_compute_energy_kwh_proxy",
    "inference_facility_energy_kwh_pue",
    "inference_compute_carbon_kg",
    "inference_mean_compute_power_kw",
    "inference_peak_compute_power_kw",
    "inference_mean_facility_power_kw",
    "inference_peak_facility_power_kw",
]

SUMMARY_OUTPUT_COLUMNS = {
    "strategy": "strategy（策略名称）",
    "objective": "objective（目标函数值；归一化加权值）",
    "compute_energy_kwh_proxy": "compute_energy_kwh_proxy（IT侧总用电量；kWh）",
    "facility_energy_kwh_pue": "facility_energy_kwh_pue（数据中心总用电量；kWh）",
    "compute_carbon_kg": "compute_carbon_kg（数据中心用电碳排放；kgCO2）",
    "tx_cost_usd": "tx_cost_usd（跨区域传输成本；USD）",
    "load_weighted_delay_hours": "load_weighted_delay_hours（负载加权平均延迟；小时）",
    "max_delay_hours": "max_delay_hours（最大延迟；小时）",
    "cpu_energy_kwh": "cpu_energy_kwh（CPU IT侧用电量；kWh）",
    "gpu_energy_kwh": "gpu_energy_kwh（GPU IT侧用电量；kWh）",
    "mem_energy_kwh": "mem_energy_kwh（内存IT侧用电量；kWh）",
    "training_compute_energy_kwh_proxy": "training_compute_energy_kwh_proxy（训练任务IT侧用电量；kWh）",
    "training_facility_energy_kwh_pue": "training_facility_energy_kwh_pue（训练任务数据中心总用电量；kWh）",
    "training_compute_carbon_kg": "training_compute_carbon_kg（训练任务数据中心用电碳排放；kgCO2）",
    "training_mean_compute_power_kw": "training_mean_compute_power_kw（训练任务平均IT功率；kW）",
    "training_peak_compute_power_kw": "training_peak_compute_power_kw（训练任务峰值IT功率；kW）",
    "training_mean_facility_power_kw": "training_mean_facility_power_kw（训练任务平均数据中心功率；kW）",
    "training_peak_facility_power_kw": "training_peak_facility_power_kw（训练任务峰值数据中心功率；kW）",
    "inference_compute_energy_kwh_proxy": "inference_compute_energy_kwh_proxy（推理任务IT侧用电量；kWh）",
    "inference_facility_energy_kwh_pue": "inference_facility_energy_kwh_pue（推理任务数据中心总用电量；kWh）",
    "inference_compute_carbon_kg": "inference_compute_carbon_kg（推理任务数据中心用电碳排放；kgCO2）",
    "inference_mean_compute_power_kw": "inference_mean_compute_power_kw（推理任务平均IT功率；kW）",
    "inference_peak_compute_power_kw": "inference_peak_compute_power_kw（推理任务峰值IT功率；kW）",
    "inference_mean_facility_power_kw": "inference_mean_facility_power_kw（推理任务平均数据中心功率；kW）",
    "inference_peak_facility_power_kw": "inference_peak_facility_power_kw（推理任务峰值数据中心功率；kW）",
    "runtime_seconds": "runtime_seconds（运行时间；秒）",
}

COUNTRY_SUMMARY_COLUMNS = [
    "strategy",
    "location",
    "compute_energy_kwh_proxy",
    "facility_energy_kwh_pue",
    "compute_carbon_kg",
    "facility_energy_share_pct",
    "compute_carbon_share_pct",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "mem_energy_kwh",
    "mean_compute_power_kw",
    "peak_compute_power_kw",
    "peak_compute_power_hour",
    "mean_facility_power_kw",
    "peak_facility_power_kw",
    "peak_facility_power_hour",
    "daily_peak_valley_facility_power_kw",
    "max_hourly_ramp_facility_power_kw",
    "high_carbon_period_facility_energy_share_pct",
    "training_compute_energy_kwh_proxy",
    "training_facility_energy_kwh_pue",
    "training_compute_carbon_kg",
    "training_mean_compute_power_kw",
    "training_peak_compute_power_kw",
    "training_mean_facility_power_kw",
    "training_peak_facility_power_kw",
    "inference_compute_energy_kwh_proxy",
    "inference_facility_energy_kwh_pue",
    "inference_compute_carbon_kg",
    "inference_mean_compute_power_kw",
    "inference_peak_compute_power_kw",
    "inference_mean_facility_power_kw",
    "inference_peak_facility_power_kw",
]

COUNTRY_SUMMARY_OUTPUT_COLUMNS = {
    "strategy": "strategy（策略名称）",
    "location": "location（国家或区域代码）",
    "compute_energy_kwh_proxy": "compute_energy_kwh_proxy（该国家IT侧总用电量；kWh）",
    "facility_energy_kwh_pue": "facility_energy_kwh_pue（该国家数据中心总用电量；kWh）",
    "compute_carbon_kg": "compute_carbon_kg（该国家数据中心用电碳排放；kgCO2）",
    "facility_energy_share_pct": "facility_energy_share_pct（该国家数据中心总用电量占24国总用电量比例；%）",
    "compute_carbon_share_pct": "compute_carbon_share_pct（该国家碳排放占24国总碳排放比例；%）",
    "cpu_energy_kwh": "cpu_energy_kwh（该国家CPU IT侧用电量；kWh）",
    "gpu_energy_kwh": "gpu_energy_kwh（该国家GPU IT侧用电量；kWh）",
    "mem_energy_kwh": "mem_energy_kwh（该国家内存IT侧用电量；kWh）",
    "mean_compute_power_kw": "mean_compute_power_kw（该国家平均IT功率；kW）",
    "peak_compute_power_kw": "peak_compute_power_kw（该国家峰值IT功率；kW）",
    "peak_compute_power_hour": "peak_compute_power_hour（该国家IT功率峰值出现小时；hour index）",
    "mean_facility_power_kw": "mean_facility_power_kw（该国家平均数据中心功率；kW）",
    "peak_facility_power_kw": "peak_facility_power_kw（该国家峰值数据中心功率；kW）",
    "peak_facility_power_hour": "peak_facility_power_hour（该国家数据中心功率峰值出现小时；hour index）",
    "daily_peak_valley_facility_power_kw": "daily_peak_valley_facility_power_kw（平均日内数据中心功率峰谷差；kW）",
    "max_hourly_ramp_facility_power_kw": "max_hourly_ramp_facility_power_kw（数据中心功率最大小时爬坡；kW）",
    "high_carbon_period_facility_energy_share_pct": "high_carbon_period_facility_energy_share_pct（高碳时段用电比例；%；高碳时段为本国本年小时碳强度前25%）",
    "training_compute_energy_kwh_proxy": "training_compute_energy_kwh_proxy（该国家训练任务IT侧用电量；kWh）",
    "training_facility_energy_kwh_pue": "training_facility_energy_kwh_pue（该国家训练任务数据中心总用电量；kWh）",
    "training_compute_carbon_kg": "training_compute_carbon_kg（该国家训练任务数据中心用电碳排放；kgCO2）",
    "training_mean_compute_power_kw": "training_mean_compute_power_kw（该国家训练任务平均IT功率；kW）",
    "training_peak_compute_power_kw": "training_peak_compute_power_kw（该国家训练任务峰值IT功率；kW）",
    "training_mean_facility_power_kw": "training_mean_facility_power_kw（该国家训练任务平均数据中心功率；kW）",
    "training_peak_facility_power_kw": "training_peak_facility_power_kw（该国家训练任务峰值数据中心功率；kW）",
    "inference_compute_energy_kwh_proxy": "inference_compute_energy_kwh_proxy（该国家推理任务IT侧用电量；kWh）",
    "inference_facility_energy_kwh_pue": "inference_facility_energy_kwh_pue（该国家推理任务数据中心总用电量；kWh）",
    "inference_compute_carbon_kg": "inference_compute_carbon_kg（该国家推理任务数据中心用电碳排放；kgCO2）",
    "inference_mean_compute_power_kw": "inference_mean_compute_power_kw（该国家推理任务平均IT功率；kW）",
    "inference_peak_compute_power_kw": "inference_peak_compute_power_kw（该国家推理任务峰值IT功率；kW）",
    "inference_mean_facility_power_kw": "inference_mean_facility_power_kw（该国家推理任务平均数据中心功率；kW）",
    "inference_peak_facility_power_kw": "inference_peak_facility_power_kw（该国家推理任务峰值数据中心功率；kW）",
}


def _empty_assignment_frame():
    return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)


def configure_runtime_parameters(args):
    global CPU_KW_PER_CORE, GPU_KW_PER_UNIT, MEM_KW_PER_GB
    global TRANSMISSION_KWH_PER_GB, POLICY_BLOCK_COST

    CPU_KW_PER_CORE = float(args.cpu_kw_per_core)
    GPU_KW_PER_UNIT = float(args.gpu_kw_per_unit)
    MEM_KW_PER_GB = float(args.mem_kw_per_gb)
    TRANSMISSION_KWH_PER_GB = float(args.transmission_kwh_per_gb)
    POLICY_BLOCK_COST = float(args.policy_block_cost)


def _it_power_kw(cores, gpu, mem):
    return cores * CPU_KW_PER_CORE + gpu * GPU_KW_PER_UNIT + mem * MEM_KW_PER_GB


def _absolute_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_numpy_pickle_compat_aliases():
    # Some workload pickles were written by NumPy versions that refer to
    # numpy._core. Keep a compatibility alias without touching deprecated
    # np.core attributes on modern NumPy.
    try:
        import numpy._core as numpy_core
        import numpy._core.multiarray as numpy_multiarray
        import numpy._core.numeric as numpy_numeric
    except Exception:
        try:
            import numpy.core as numpy_core
            import numpy.core.multiarray as numpy_multiarray
            import numpy.core.numeric as numpy_numeric
        except Exception:
            numpy_core = None
            numpy_multiarray = None
            numpy_numeric = None
    if numpy_core is not None:
        sys.modules.setdefault("numpy._core", numpy_core)
    if numpy_multiarray is not None:
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    if numpy_numeric is not None:
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def _factor_scalar(value, factor_index):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) == 0:
        raise ValueError("Empty factor value.")
    if len(arr) == 1:
        return float(arr[0])
    if factor_index < 0 or factor_index >= len(arr):
        raise ValueError(f"Factor index {factor_index} out of range for value with length {len(arr)}.")
    return float(arr[factor_index])


def _pue_scalar(value, factor_index, scenario_index):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return _factor_scalar(arr, factor_index)
    if arr.ndim != 2:
        raise ValueError(f"PUE value must be scalar, 1-D, or 2-D; got shape {arr.shape}.")
    if factor_index < 0 or factor_index >= arr.shape[0]:
        raise ValueError(f"PUE year index {factor_index} out of range for shape {arr.shape}.")
    if scenario_index < 0 or scenario_index >= arr.shape[1]:
        raise ValueError(f"PUE scenario index {scenario_index} out of range for shape {arr.shape}.")
    return float(arr[factor_index, scenario_index])


def _llm_dlc_rate(index, initial=0.05, growth=0.2):
    rate = float(initial)
    for _ in range(int(index)):
        rate *= 1.0 + float(growth)
    return rate


def build_water_arrays(dc_cfg, water_path, water_index, n_hours, apply_dlc_adjustment=True, pue_scenario="Base", grid_water_policy="CP"):
    module = _load_module_from_path("oracle_water_factors", _absolute_path(water_path))
    if not hasattr(module, "WUE_BY_LOCATION"):
        raise ValueError(f"{water_path} must define WUE_BY_LOCATION.")
    if not hasattr(module, "PUE_BY_LOCATION"):
        raise ValueError(f"{water_path} must define PUE_BY_LOCATION.")

    grid_water_policy = str(grid_water_policy).upper()
    grid_water_attr = f"GRID_WATER_FACTORS_{grid_water_policy}_BY_LOCATION"
    if not hasattr(module, grid_water_attr):
        raise ValueError(f"{water_path} must define {grid_water_attr}.")

    wue_by_location = module.WUE_BY_LOCATION
    pue_by_location = module.PUE_BY_LOCATION
    grid_water_by_location = getattr(module, grid_water_attr)
    pue_scenarios = list(getattr(module, "PUE_SCENARIOS", ["Base"]))
    if pue_scenario not in pue_scenarios:
        raise ValueError(f"Unknown PUE scenario '{pue_scenario}'. Available: {pue_scenarios}.")
    pue_scenario_index = pue_scenarios.index(pue_scenario)
    direct_wue = np.zeros((n_hours, len(dc_cfg)), dtype=float)
    grid_water = np.zeros((n_hours, len(dc_cfg)), dtype=float)
    pue = np.ones((n_hours, len(dc_cfg)), dtype=float)
    dlc_rate = _llm_dlc_rate(water_index)

    for loc_idx, dc in enumerate(dc_cfg):
        location = dc["location"]
        if location not in wue_by_location:
            raise ValueError(f"Missing WUE for location '{location}' in {water_path}.")
        if location not in pue_by_location:
            raise ValueError(f"Missing PUE for location '{location}' in {water_path}.")
        if location not in grid_water_by_location:
            raise ValueError(f"Missing grid water factor for location '{location}' in {water_path}.")

        base_wue = _factor_scalar(wue_by_location[location], water_index)
        base_grid_water = _factor_scalar(grid_water_by_location[location], water_index)
        if apply_dlc_adjustment:
            base_wue = base_wue * (1.0 - dlc_rate) + (base_wue - 0.137) * dlc_rate
        direct_wue[:, loc_idx] = base_wue
        grid_water[:, loc_idx] = base_grid_water
        pue[:, loc_idx] = _pue_scalar(pue_by_location[location], water_index, pue_scenario_index)

    return direct_wue, grid_water, pue


def _fit_length(values, n_hours):
    values = np.asarray(values, dtype=float)
    if len(values) >= n_hours:
        return values[:n_hours]
    if len(values) == 0:
        return np.zeros(n_hours, dtype=float)
    return np.resize(values, n_hours)


def _load_hourly_ci(location, year, timezone_shift, n_hours, carbon_data_root=None, carbon_policy=None):
    if carbon_data_root:
        try:
            return _load_em_hourly_ci(
                location,
                year,
                timezone_shift,
                n_hours,
                carbon_data_root=carbon_data_root,
                carbon_policy=carbon_policy,
            )
        except FileNotFoundError:
            pass

    path = _absolute_path(
        f"data/carbon_intensity/{location}/{year}/{location}_{year}_hourly.csv"
    )
    if not os.path.exists(path):
        return _load_em_hourly_ci(location, year, timezone_shift, n_hours)
    df = pd.read_csv(path)
    value_col = next(
        (col for col in df.columns if "Carbon Intensity" in col and "direct" in col),
        None,
    )
    if value_col is None:
        raise ValueError(f"Missing direct carbon intensity column in {path}.")
    values = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    # CI_Manager effectively shifts the interpolated array by timezone_shift.
    values = np.roll(values, -int(timezone_shift))
    return _fit_length(values, n_hours)


def _load_em_hourly_ci(location, year, timezone_shift, n_hours, carbon_data_root="data/EM-download", carbon_policy=None):
    folder_name = "Great Britain" if location == "United_Kingdom" else location
    root = carbon_data_root or "data/EM-download"
    folder = _absolute_path(os.path.join(root, folder_name))
    policy = str(carbon_policy).upper() if carbon_policy else None
    suffix = f"-{policy}-{int(year)}-hourly.csv" if policy else f"-{int(year)}-hourly.csv"
    pattern = f"*{suffix}"
    candidates = sorted(
        path for path in (os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(suffix))
        if os.path.basename(path).endswith(suffix)
    ) if os.path.isdir(folder) else []
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple carbon intensity files for '{location}' year={year}, "
            f"policy={policy or 'none'} under {folder}: {candidates}"
        )
    if not candidates:
        raise FileNotFoundError(
            f"Missing carbon intensity for '{location}'. Tried "
            f"data/carbon_intensity/{location}/{year}/{location}_{year}_hourly.csv "
            f"and {os.path.join(folder, pattern)}."
        )

    df = pd.read_csv(candidates[0])
    value_col = next(
        (col for col in df.columns if "Carbon intensity" in col and "direct" in col),
        None,
    )
    if value_col is None:
        value_col = next(
            (col for col in df.columns if "Carbon Intensity" in col and "direct" in col),
            None,
        )
    if value_col is None:
        raise ValueError(f"Missing direct carbon intensity column in {candidates[0]}.")
    timestamps = pd.to_datetime(df["Datetime (UTC)"], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=timestamps).sort_index()
    series = series[~series.index.isna()]
    series = series[~series.index.duplicated(keep="first")]
    full_index = pd.date_range(
        f"{int(year)}-01-01 00:00:00",
        periods=8760,
        freq="h",
    )
    series = series.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()
    values = series.to_numpy(dtype=float)
    values = np.roll(values, -int(timezone_shift))
    return _fit_length(values, n_hours)


def build_carbon_arrays(dc_cfg, year, n_hours, carbon_data_root="data/EM-estimate", carbon_policy="CP"):
    locations = [dc["location"] for dc in dc_cfg]
    ci = np.zeros((n_hours, len(locations)), dtype=float)
    for loc_idx, dc in enumerate(dc_cfg):
        timezone_shift = int(dc.get("timezone_shift", 0))
        ci[:, loc_idx] = _load_hourly_ci(
            dc["location"],
            year,
            timezone_shift,
            n_hours,
            carbon_data_root=carbon_data_root,
            carbon_policy=carbon_policy,
        )
    return ci


def _percentile_rank(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


def _llm_task_types_from_arrays(task_ids, duration_minutes, cores_req, gpu_req, gpu_mem_req, mem_req, bandwidth_gb, training_ratio):
    if not (0.0 <= float(training_ratio) <= 1.0):
        raise ValueError(f"training_ratio must be in [0, 1], got {training_ratio}")

    task_ids = np.asarray(task_ids, dtype=int)
    duration_minutes = np.asarray(duration_minutes, dtype=float)
    cores_req = np.asarray(cores_req, dtype=float)
    gpu_req = np.asarray(gpu_req, dtype=float)
    gpu_mem_req = np.asarray(gpu_mem_req, dtype=float)
    mem_req = np.asarray(mem_req, dtype=float)
    bandwidth_gb = np.asarray(bandwidth_gb, dtype=float)
    n_tasks = len(task_ids)
    n_training = int(round(n_tasks * float(training_ratio)))

    duration_hours = np.maximum(duration_minutes, 0.0) / 60.0
    gpu_hours = gpu_req * duration_hours
    gpu_mem_hours = gpu_mem_req * duration_hours
    compute_proxy = _it_power_kw(cores_req, gpu_req, mem_req) * duration_hours
    bandwidth_per_compute = bandwidth_gb / (compute_proxy + 1e-9)

    duration_rank = _percentile_rank(duration_minutes)
    gpu_hours_rank = _percentile_rank(gpu_hours)
    gpu_mem_hours_rank = _percentile_rank(gpu_mem_hours)
    compute_rank = _percentile_rank(compute_proxy)
    bandwidth_per_compute_rank = _percentile_rank(bandwidth_per_compute)

    training_score = (
        0.40 * gpu_hours_rank
        + 0.25 * gpu_mem_hours_rank
        + 0.20 * duration_rank
        + 0.15 * compute_rank
    )
    inference_score = (
        0.35 * (1.0 - duration_rank)
        + 0.30 * (1.0 - gpu_hours_rank)
        + 0.20 * (1.0 - compute_rank)
        + 0.15 * bandwidth_per_compute_rank
    )
    training_likeness = training_score - inference_score

    order = np.lexsort((-task_ids, bandwidth_gb, training_likeness))
    task_types = np.full(n_tasks, "inference", dtype=object)
    if n_training > 0:
        task_types[order[-n_training:]] = "training"
    return task_types


def _finalize_hourly_aggregate_type(record, task_type, n_hours):
    defaults = TASK_TYPE_DEFAULTS[task_type]
    record["task_type"] = task_type
    record["run_hours"] = 1

    if task_type == "inference":
        record["latest_start_hour"] = int(record["arrival_hour"])
        record["migration_allowed"] = False
        return record

    if defaults["defer_allowed"]:
        window_hours = int(math.ceil(defaults["max_delay_minutes"] / 60.0))
        latest_start = min(int(record["arrival_hour"]) + window_hours, int(n_hours) - 1)
    else:
        latest_start = int(record["arrival_hour"])
    record["latest_start_hour"] = int(latest_start)
    record["migration_allowed"] = bool(defaults["migration_allowed"])
    return record


def _load_share_weights(dc_cfg, load_share_mode):
    weights = []
    for dc in dc_cfg:
        weights.append(float(dc.get(load_share_mode, dc.get("it_ratio_weitgh", 1.0))))

    shares = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(shares)):
        raise ValueError(f"{load_share_mode} contains non-finite values.")
    total = float(shares.sum())
    if total <= 0.0:
        raise ValueError(f"Total {load_share_mode} must be positive.")
    return shares / total


def load_alibaba_hourly_aggregate_tasks(workload_path, dc_cfg, task_scale, year, n_hours, llm_training_ratio, load_share_mode, max_tasks=None):
    del year
    _install_numpy_pickle_compat_aliases()
    path = _absolute_path(workload_path)
    df = pd.read_pickle(path)
    if "interval_15m" in df.columns:
        df = df.sort_values("interval_15m")
    df = df.head(n_hours * 4).copy()

    raw_arrival_hour = []
    raw_duration = []
    raw_cores = []
    raw_gpu = []
    raw_gpu_mem = []
    raw_mem = []
    raw_bandwidth = []
    for interval_idx, (_, row) in enumerate(df.iterrows()):
        arrival_hour = interval_idx // 4
        if arrival_hour >= n_hours:
            break
        for task_data in row["tasks_matrix"]:
            duration = float(task_data[4])
            duration_hours = max(duration, 0.0) / 60.0
            cores_req = float(task_scale) * float(task_data[5]) / 100.0
            gpu_req = float(task_scale) * float(task_data[6]) / 100.0
            mem_req = float(task_scale) * float(task_data[7])
            gpu_mem_req = float(task_scale) * float(task_data[8])
            # bandwidth_gb = float(task_data[9])
            bandwidth_gb = float(task_scale) * float(task_data[9])
            raw_arrival_hour.append(int(arrival_hour))
            raw_duration.append(duration)
            raw_cores.append(max(cores_req, 0.0))
            raw_gpu.append(max(gpu_req, 0.0))
            raw_gpu_mem.append(max(gpu_mem_req, 0.0))
            raw_mem.append(max(mem_req, 0.0))
            raw_bandwidth.append(max(bandwidth_gb, 0.0))

    raw_task_type = _llm_task_types_from_arrays(
        task_ids=np.arange(len(raw_arrival_hour), dtype=int),
        duration_minutes=raw_duration,
        cores_req=raw_cores,
        gpu_req=raw_gpu,
        gpu_mem_req=raw_gpu_mem,
        mem_req=raw_mem,
        bandwidth_gb=raw_bandwidth,
        training_ratio=llm_training_ratio,
    )

    buckets = {}
    raw_arrival_hour = np.asarray(raw_arrival_hour, dtype=np.int32)
    raw_duration_hours = np.maximum(np.asarray(raw_duration, dtype=float), 0.0) / 60.0
    raw_cores = np.asarray(raw_cores, dtype=float)
    raw_gpu = np.asarray(raw_gpu, dtype=float)
    raw_gpu_mem = np.asarray(raw_gpu_mem, dtype=float)
    raw_mem = np.asarray(raw_mem, dtype=float)
    raw_bandwidth = np.asarray(raw_bandwidth, dtype=float)
    for idx, task_type in enumerate(raw_task_type):
        key = (int(raw_arrival_hour[idx]), str(task_type))
        if key not in buckets:
            buckets[key] = np.zeros(6, dtype=float)
        buckets[key] += np.asarray(
            [
                raw_cores[idx] * raw_duration_hours[idx],
                raw_gpu[idx] * raw_duration_hours[idx],
                raw_gpu_mem[idx] * raw_duration_hours[idx],
                raw_mem[idx] * raw_duration_hours[idx],
                raw_bandwidth[idx],
                1.0,
            ],
            dtype=float,
        )

    shares = _load_share_weights(dc_cfg, load_share_mode)
    records = []
    for (arrival_hour, task_type), resources in sorted(buckets.items()):
        if np.all(resources[:5] <= 1e-12):
            continue
        for origin_idx, dc in enumerate(dc_cfg):
            share = float(shares[origin_idx])
            record = {
                "task_id": len(records),
                "job_name": f"hourly_aggregate_{arrival_hour}_{task_type}_{origin_idx}",
                "resource_task_type": task_type,
                "arrival_hour": int(arrival_hour),
                "duration_minutes": 60.0,
                "cores_req": float(resources[0] * share),
                "gpu_req": float(resources[1] * share),
                "gpu_mem_req": float(resources[2] * share),
                "mem_req": float(resources[3] * share),
                "bandwidth_gb": float(resources[4] * share),
                "source_task_count": float(resources[5] * share),
                "origin_idx": origin_idx,
                "origin": dc["location"],
            }
            _finalize_hourly_aggregate_type(record, task_type, n_hours)
            records.append(record)
            if max_tasks is not None and len(records) >= max_tasks:
                return pd.DataFrame(records)
    return pd.DataFrame(records)


def filter_tasks_schedulable_within_horizon(tasks, n_hours):
    if tasks.empty:
        return tasks, 0
    latest_start = np.minimum(
        tasks["latest_start_hour"].to_numpy(dtype=int),
        int(n_hours) - tasks["run_hours"].to_numpy(dtype=int),
    )
    arrivals = tasks["arrival_hour"].to_numpy(dtype=int)
    schedulable = latest_start >= arrivals
    dropped = int((~schedulable).sum())
    if dropped == 0:
        return tasks, 0
    return tasks.loc[schedulable].copy(), dropped


def _load_country_square_frame(path, locations, label):
    matrix_path = _absolute_path(path)
    matrix = pd.read_csv(matrix_path, index_col=0)
    missing_rows = [location for location in locations if location not in matrix.index]
    missing_cols = [location for location in locations if location not in matrix.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"{label} must contain all datacenter locations. "
            f"Missing rows={missing_rows}; missing columns={missing_cols}; path={matrix_path}"
        )
    return matrix.loc[list(locations), list(locations)]


def _transmission_cost_matrix(locations, cloud_provider, policy_matrix_path=None):
    if policy_matrix_path:
        return _load_country_square_frame(
            policy_matrix_path,
            locations,
            "Policy transmission matrix",
        ).to_numpy(dtype=float)
    matrix = load_transmission_matrix(cloud_provider)
    out = np.zeros((len(locations), len(locations)), dtype=float)
    for origin_idx, origin in enumerate(locations):
        origin_region = map_location_to_region(origin, cloud_provider)
        for dest_idx, destination in enumerate(locations):
            dest_region = map_location_to_region(destination, cloud_provider)
            out[origin_idx, dest_idx] = float(matrix.loc[origin_region, dest_region])
    return out


def _policy_allowed_mask(locations, policy_allowed_mask_path=None):
    mask = np.ones((len(locations), len(locations)), dtype=bool)
    if not policy_allowed_mask_path:
        return mask
    raw = _load_country_square_frame(
        policy_allowed_mask_path,
        locations,
        "Policy allowed mask",
    )
    normalized = raw.replace(
        {
            True: 1,
            False: 0,
            "true": 1,
            "false": 0,
            "True": 1,
            "False": 0,
            "yes": 1,
            "no": 0,
            "YES": 1,
            "NO": 0,
        }
    )
    mask = normalized.to_numpy(dtype=float) > 0.5
    np.fill_diagonal(mask, True)
    return mask


def _allowed_mask_from_policy_costs(tx_cost_per_gb, policy_allowed_mask=None):
    cost_mask = np.asarray(tx_cost_per_gb, dtype=float) < POLICY_BLOCK_COST
    np.fill_diagonal(cost_mask, True)
    if policy_allowed_mask is None:
        return cost_mask
    mask = np.asarray(policy_allowed_mask, dtype=bool) & cost_mask
    np.fill_diagonal(mask, True)
    return mask


def _delay_hours(origin, destination, cloud_provider, bandwidth_gb, mode):
    return 0


def _task_it_power_kw(row):
    return _it_power_kw(float(row["cores_req"]), float(row["gpu_req"]), float(row["mem_req"]))


def _load_it_capacity_ratio_data(path):
    module = _load_module_from_path("oracle_it_capacity_ratios", _absolute_path(path))
    if not hasattr(module, "IT_RATIO_BY_LOCATION"):
        raise ValueError(f"{path} must define IT_RATIO_BY_LOCATION.")
    ratios = dict(module.IT_RATIO_BY_LOCATION)
    aliases = dict(getattr(module, "IT_RATIO_LOCATION_ALIASES", {}))
    return ratios, aliases


def _resolve_it_ratio_location(dc, ratios, aliases):
    location = dc["location"]
    ratio_location = dc.get("it_ratio_location", aliases.get(location, location))
    if ratio_location not in ratios:
        raise ValueError(
            f"No IT capacity ratio for location '{location}' "
            f"(resolved as '{ratio_location}'). Add it to the ratio file or set "
            f"`it_ratio_location` in the datacenter config."
        )
    return ratio_location


def _alibaba_peak_required_total_it_capacity_kw(tasks, weights, n_locations):
    hourly_power = np.zeros((int(tasks["arrival_hour"].max()) + 1, n_locations), dtype=float)
    arrival_hours = tasks["arrival_hour"].to_numpy(dtype=np.int64)
    origin_indices = tasks["origin_idx"].to_numpy(dtype=np.int64)
    task_power = _it_power_kw(
        tasks["cores_req"].to_numpy(dtype=float),
        tasks["gpu_req"].to_numpy(dtype=float),
        tasks["mem_req"].to_numpy(dtype=float),
    )
    np.add.at(hourly_power, (arrival_hours, origin_indices), task_power)
    if hourly_power.size == 0:
        return 0.0
    weights = np.asarray(weights, dtype=float).reshape(-1)
    positive = weights > 0
    if not np.all(positive):
        raise ValueError("IT ratio weights must be positive when using alibaba_peak.")
    return float(np.max(hourly_power[:, positive] / weights[positive]))


def build_fixed_it_capacity_kw(dc_cfg, tasks, ratio_path, utilization, capacity_multiplier=1.0):
    ratios, aliases = _load_it_capacity_ratio_data(ratio_path)
    ratio_locations = [_resolve_it_ratio_location(dc, ratios, aliases) for dc in dc_cfg]
    weights = np.asarray([float(ratios[ratio_location]) for ratio_location in ratio_locations], dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("IT capacity ratios must be non-negative and have positive sum for selected locations.")
    weights = weights / float(weights.sum())

    if not (0.0 < float(utilization) <= 1.0):
        raise ValueError("--it-capacity-utilization must be in (0, 1].")
    total_capacity_kw = _alibaba_peak_required_total_it_capacity_kw(tasks, weights, len(dc_cfg)) / float(utilization)

    if total_capacity_kw <= 0:
        raise ValueError("Computed total IT capacity must be positive.")
    total_capacity_kw *= float(capacity_multiplier)
    return total_capacity_kw * weights, ratio_locations, float(total_capacity_kw)


def _task_capacity_vector(task):
    return np.asarray([_task_it_power_kw(task)], dtype=float)


def _capacity_matrix(dc_cfg, it_capacity_kw):
    values = np.asarray(it_capacity_kw, dtype=float).reshape(-1)
    if len(values) != len(dc_cfg):
        raise ValueError("it_capacity_kw length must match the number of datacenters.")
    if np.any(values < 0):
        raise ValueError("it_capacity_kw values must be non-negative.")
    return values.reshape(len(dc_cfg), 1)


def _task_resource_matrix(tasks):
    cores = tasks["cores_req"].to_numpy(dtype=float)
    gpu = tasks["gpu_req"].to_numpy(dtype=float)
    mem = tasks["mem_req"].to_numpy(dtype=float)
    return _it_power_kw(cores, gpu, mem).reshape(-1, 1)


def _task_rows_for_options(tasks, option_task):
    task_ids = tasks["task_id"].to_numpy(dtype=np.int64)
    option_task = np.asarray(option_task, dtype=np.int64)
    if len(option_task) == 0:
        return np.asarray([], dtype=np.int32)
    if np.array_equal(task_ids, np.arange(len(task_ids), dtype=np.int64)):
        return option_task.astype(np.int32, copy=False)
    task_to_row = {int(task_id): row_idx for row_idx, task_id in enumerate(task_ids)}
    return np.fromiter((task_to_row[int(task_id)] for task_id in option_task), dtype=np.int32, count=len(option_task))


def _concat_option_parts(parts, dtype):
    if not parts:
        return np.asarray([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _progress(message, start_time=None):
    if start_time is None:
        print(f"[progress] {message}", flush=True)
        return
    print(f"[progress] {message} elapsed={time.perf_counter() - start_time:.1f}s", flush=True)


def _new_option_parts():
    return {field: [] for field in OPTION_FIELD_DTYPES}


def _concat_option_part_dict(parts):
    return {
        field: _concat_option_parts(parts[field], OPTION_FIELD_DTYPES[field])
        for field in OPTION_FIELD_DTYPES
    }


def _option_data_from_arrays(arrays, task_option_groups, task_id_to_group=None):
    option_task_array = arrays["option_task"]
    return OptionData(
        option_task=option_task_array,
        option_start=arrays["option_start"],
        option_finish=arrays["option_finish"],
        option_dest=arrays["option_dest"],
        option_tx_delay_hours=arrays["option_tx_delay_hours"],
        option_pue=arrays["option_pue"],
        option_compute_energy_kwh=arrays["option_compute_energy_kwh"],
        option_facility_energy_kwh=arrays["option_facility_energy_kwh"],
        option_compute_carbon_kg=arrays["option_compute_carbon_kg"],
        option_direct_water_m3=arrays["option_direct_water_m3"],
        option_grid_water_m3=arrays["option_grid_water_m3"],
        option_water_m3=arrays["option_water_m3"],
        option_transmission_energy_kwh=arrays["option_transmission_energy_kwh"],
        option_transmission_carbon_kg=arrays["option_transmission_carbon_kg"],
        option_carbon_kg=arrays["option_carbon_kg"],
        option_tx_cost_usd=arrays["option_tx_cost_usd"],
        option_objective=np.zeros(len(option_task_array), dtype=float),
        task_option_groups=task_option_groups,
        task_id_to_group={} if task_id_to_group is None else task_id_to_group,
    )


def _option_data_from_parts(parts, task_option_groups, task_id_to_group=None):
    return _option_data_from_arrays(
        _concat_option_part_dict(parts),
        task_option_groups,
        task_id_to_group=task_id_to_group,
    )


def _make_option_build_context(tasks, dc_cfg, ci_by_hour_loc, direct_wue_by_hour_loc, grid_water_by_hour_loc, pue_by_hour_loc, cloud_provider, include_transmission_carbon, delay_rounding, n_hours, policy_transmission_matrix_path=None, policy_allowed_mask_path=None):
    locations = tuple(dc["location"] for dc in dc_cfg)
    duration_hours = np.maximum(tasks["duration_minutes"].to_numpy(dtype=float), 0.0) / 60.0
    compute_energy_values = _it_power_kw(
        tasks["cores_req"].to_numpy(dtype=float),
        tasks["gpu_req"].to_numpy(dtype=float),
        tasks["mem_req"].to_numpy(dtype=float),
    ) * duration_hours
    tx_cost_per_gb = _transmission_cost_matrix(
        locations,
        cloud_provider,
        policy_matrix_path=policy_transmission_matrix_path,
    )
    allowed_dest_mask = _allowed_mask_from_policy_costs(
        tx_cost_per_gb,
        _policy_allowed_mask(locations, policy_allowed_mask_path),
    )
    return {
        "locations": locations,
        "tx_cost_per_gb": tx_cost_per_gb,
        "allowed_dest_mask": allowed_dest_mask,
        "ci_by_hour_loc": ci_by_hour_loc,
        "direct_wue_by_hour_loc": direct_wue_by_hour_loc,
        "grid_water_by_hour_loc": grid_water_by_hour_loc,
        "pue_by_hour_loc": pue_by_hour_loc,
        "cloud_provider": cloud_provider,
        "include_transmission_carbon": bool(include_transmission_carbon),
        "delay_rounding": delay_rounding,
        "n_hours": int(n_hours),
        "task_ids": tasks["task_id"].to_numpy(dtype=np.int32),
        "origin_indices": tasks["origin_idx"].to_numpy(dtype=np.int32),
        "arrival_hours": tasks["arrival_hour"].to_numpy(dtype=np.int32),
        "latest_start_hours": tasks["latest_start_hour"].to_numpy(dtype=np.int32),
        "migration_allowed": tasks["migration_allowed"].to_numpy(dtype=bool),
        "bandwidth_values": tasks["bandwidth_gb"].to_numpy(dtype=float),
        "compute_energy_values": compute_energy_values,
        "task_types": tasks["task_type"].to_numpy(dtype=object),
        "total_tasks": len(tasks),
        "transmission_kwh_per_gb": TRANSMISSION_KWH_PER_GB,
    }


def _append_option_arrays(parts, arrays, selected=None):
    for field in OPTION_FIELD_DTYPES:
        values = arrays[field]
        if selected is not None:
            values = values[selected]
        parts[field].append(values)


def _build_one_hour_option_chunk(context, start_row, end_row, show_progress=False):
    locations = context["locations"]
    tx_cost_per_gb = context["tx_cost_per_gb"]
    allowed_dest_mask = context["allowed_dest_mask"]
    ci_by_hour_loc = context["ci_by_hour_loc"]
    direct_wue_by_hour_loc = context["direct_wue_by_hour_loc"]
    grid_water_by_hour_loc = context["grid_water_by_hour_loc"]
    pue_by_hour_loc = context["pue_by_hour_loc"]
    cloud_provider = context["cloud_provider"]
    include_transmission_carbon = context["include_transmission_carbon"]
    delay_rounding = context["delay_rounding"]
    n_hours = context["n_hours"]
    task_ids = context["task_ids"]
    origin_indices = context["origin_indices"]
    arrival_hours = context["arrival_hours"]
    latest_start_hours = context["latest_start_hours"]
    migration_allowed = context["migration_allowed"]
    bandwidth_values = context["bandwidth_values"]
    compute_energy_values = context["compute_energy_values"]
    task_types = context["task_types"]
    transmission_kwh_per_gb = float(context["transmission_kwh_per_gb"])

    parts = _new_option_parts()
    task_option_groups = []
    cursor = 0
    delay_cache = {}
    n_locations = len(locations)
    total_tasks = int(context["total_tasks"])
    progress_step = max(1, total_tasks // 20)
    progress_start = time.perf_counter()
    if show_progress:
        _progress(f"Building options: 0/{total_tasks} tasks, 0 options")

    for row_idx in range(int(start_row), int(end_row)):
        group_start = cursor
        task_id = int(task_ids[row_idx])
        origin_idx = int(origin_indices[row_idx])
        arrival_hour = int(arrival_hours[row_idx])
        latest_start = min(int(latest_start_hours[row_idx]), n_hours - 1)
        if bool(migration_allowed[row_idx]):
            destinations = np.flatnonzero(allowed_dest_mask[origin_idx])
        else:
            destinations = [origin_idx]
        compute_energy_kwh = float(compute_energy_values[row_idx])
        bandwidth_gb = float(bandwidth_values[row_idx])
        unpruned_group_count = 0

        for dest_idx in destinations:
            delay_key = (origin_idx, dest_idx, bandwidth_gb, delay_rounding)
            if delay_key not in delay_cache:
                delay_cache[delay_key] = _delay_hours(
                    locations[origin_idx], locations[dest_idx], cloud_provider, bandwidth_gb, delay_rounding
                )
            tx_delay = delay_cache[delay_key]
            earliest_start = max(arrival_hour + tx_delay, 0)
            if latest_start < earliest_start:
                continue

            starts = np.arange(earliest_start, latest_start + 1, dtype=np.int32)
            n_options = len(starts)
            avg_ci = ci_by_hour_loc[starts, dest_idx]
            avg_direct_wue = direct_wue_by_hour_loc[starts, dest_idx]
            avg_grid_water = grid_water_by_hour_loc[starts, dest_idx]
            avg_pue = pue_by_hour_loc[starts, dest_idx]
            facility_energy_kwh = compute_energy_kwh * avg_pue
            facility_energy_mwh = facility_energy_kwh / 1000.0
            direct_water = facility_energy_mwh * avg_direct_wue
            grid_water = facility_energy_mwh * avg_grid_water
            total_water = direct_water + grid_water
            compute_carbon = facility_energy_kwh * avg_ci / 1000.0

            tx_energy = 0.0
            tx_carbon = 0.0
            tx_cost = 0.0
            if dest_idx != origin_idx:
                tx_energy = bandwidth_gb * transmission_kwh_per_gb
                origin_ci = float(ci_by_hour_loc[arrival_hour, origin_idx]) / 1000.0
                tx_carbon = tx_energy * origin_ci
                tx_cost = tx_cost_per_gb[origin_idx, dest_idx] * bandwidth_gb
            total_carbon = compute_carbon + (tx_carbon if include_transmission_carbon else 0.0)

            option_arrays = {
                "option_task": np.full(n_options, task_id, dtype=np.int32),
                "option_start": starts.astype(np.int16, copy=False),
                "option_finish": (starts + 1).astype(np.int16, copy=False),
                "option_dest": np.full(n_options, dest_idx, dtype=np.int16),
                "option_tx_delay_hours": np.full(n_options, tx_delay, dtype=np.int16),
                "option_pue": avg_pue.astype(float, copy=False),
                "option_compute_energy_kwh": np.full(n_options, compute_energy_kwh, dtype=float),
                "option_facility_energy_kwh": facility_energy_kwh.astype(float, copy=False),
                "option_compute_carbon_kg": compute_carbon.astype(float, copy=False),
                "option_direct_water_m3": direct_water.astype(float, copy=False),
                "option_grid_water_m3": grid_water.astype(float, copy=False),
                "option_water_m3": total_water.astype(float, copy=False),
                "option_transmission_energy_kwh": np.full(n_options, tx_energy, dtype=float),
                "option_transmission_carbon_kg": np.full(n_options, tx_carbon, dtype=float),
                "option_carbon_kg": total_carbon.astype(float, copy=False),
                "option_tx_cost_usd": np.full(n_options, tx_cost, dtype=float),
            }
            _append_option_arrays(parts, option_arrays)
            unpruned_group_count += n_options
            cursor += n_options

        if unpruned_group_count == 0:
            raise ValueError(
                f"No legal execution option for task_id={task_id} "
                f"arrival={arrival_hour} type={task_types[row_idx]}."
            )

        task_option_groups.append(np.arange(group_start, cursor, dtype=np.int32))
        if show_progress and ((row_idx + 1) % progress_step == 0 or row_idx + 1 == total_tasks):
            pct = 100.0 * (row_idx + 1) / max(total_tasks, 1)
            _progress(
                f"Building options: {row_idx + 1}/{total_tasks} tasks "
                f"({pct:.1f}%), options={cursor}",
                progress_start,
            )

    return _option_data_from_parts(parts, task_option_groups)


def _init_option_worker(context):
    global _OPTION_WORKER_CONTEXT
    _OPTION_WORKER_CONTEXT = context


def _build_option_chunk_worker(job):
    chunk_idx, start_row, end_row = job
    return chunk_idx, _build_one_hour_option_chunk(_OPTION_WORKER_CONTEXT, start_row, end_row)


def _option_chunk_ranges(total_tasks, workers, option_chunk_size):
    if total_tasks <= 0:
        return []
    if option_chunk_size is None or int(option_chunk_size) <= 0:
        target_chunks = max(1, int(workers) * 4)
        option_chunk_size = max(1, int(math.ceil(total_tasks / target_chunks)))
    option_chunk_size = max(1, int(option_chunk_size))
    return [
        (start, min(start + option_chunk_size, total_tasks))
        for start in range(0, total_tasks, option_chunk_size)
    ]


def _resolve_option_workers(option_workers, total_tasks):
    if total_tasks <= 1:
        return 1
    if option_workers is None or int(option_workers) <= 0:
        cpu_count = os.cpu_count() or 1
        option_workers = max(1, cpu_count - 1) if cpu_count > 1 else 1
    return max(1, min(int(option_workers), int(total_tasks)))


def _multiprocessing_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _combine_option_chunks(tasks, chunks):
    parts = _new_option_parts()
    task_option_groups = []
    option_offset = 0
    for chunk in chunks:
        for field in OPTION_FIELD_DTYPES:
            parts[field].append(getattr(chunk, field))
        for group in chunk.task_option_groups:
            task_option_groups.append((group + option_offset).astype(np.int32, copy=False))
        option_offset += len(chunk.option_task)

    task_id_to_group = {
        int(task_id): task_option_groups[row_idx]
        for row_idx, task_id in enumerate(tasks["task_id"].to_numpy(dtype=int))
    }
    return _option_data_from_parts(parts, task_option_groups, task_id_to_group=task_id_to_group)



def _build_one_hour_options(tasks, dc_cfg, ci_by_hour_loc, direct_wue_by_hour_loc, grid_water_by_hour_loc, pue_by_hour_loc, cloud_provider, include_transmission_carbon, delay_rounding, n_hours, option_workers=1, option_chunk_size=0, policy_transmission_matrix_path=None, policy_allowed_mask_path=None):
    progress_start = time.perf_counter()
    total_tasks = len(tasks)
    context = _make_option_build_context(
        tasks,
        dc_cfg,
        ci_by_hour_loc,
        direct_wue_by_hour_loc,
        grid_water_by_hour_loc,
        pue_by_hour_loc,
        cloud_provider,
        include_transmission_carbon,
        delay_rounding,
        n_hours,
        policy_transmission_matrix_path=policy_transmission_matrix_path,
        policy_allowed_mask_path=policy_allowed_mask_path,
    )
    workers = _resolve_option_workers(option_workers, total_tasks)
    chunks = _option_chunk_ranges(total_tasks, workers, option_chunk_size)
    workers = min(workers, max(len(chunks), 1))

    if workers <= 1 or len(chunks) <= 1:
        result = _build_one_hour_option_chunk(context, 0, total_tasks, show_progress=True)
        task_id_to_group = {
            int(task_id): result.task_option_groups[row_idx]
            for row_idx, task_id in enumerate(tasks["task_id"].to_numpy(dtype=int))
        }
        result.task_id_to_group = task_id_to_group
        return result

    _progress(
        f"Building options in parallel: tasks={total_tasks}, "
        f"workers={workers}, chunks={len(chunks)}"
    )
    results = [None] * len(chunks)
    completed_chunks = 0
    completed_tasks = 0
    completed_options = 0
    progress_step = max(1, len(chunks) // 20)
    jobs = [(idx, start, end) for idx, (start, end) in enumerate(chunks)]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_multiprocessing_context(),
        initializer=_init_option_worker,
        initargs=(context,),
    ) as executor:
        futures = {executor.submit(_build_option_chunk_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            chunk_idx, chunk_result = future.result()
            results[chunk_idx] = chunk_result
            completed_chunks += 1
            completed_tasks += int(job[2]) - int(job[1])
            completed_options += len(chunk_result.option_task)
            if completed_chunks % progress_step == 0 or completed_chunks == len(chunks):
                pct = 100.0 * completed_tasks / max(total_tasks, 1)
                _progress(
                    f"Building options: {completed_tasks}/{total_tasks} tasks "
                    f"({pct:.1f}%), chunks={completed_chunks}/{len(chunks)}, "
                    f"options={completed_options}",
                    progress_start,
                )

    combined = _combine_option_chunks(tasks, results)
    _progress(f"Combined option chunks: options={len(combined.option_task)}", progress_start)
    return combined


def build_options(tasks, dc_cfg, ci_by_hour_loc, direct_wue_by_hour_loc, grid_water_by_hour_loc, pue_by_hour_loc, cloud_provider, include_transmission_carbon, delay_rounding, n_hours, option_workers=1, option_chunk_size=0, policy_transmission_matrix_path=None, policy_allowed_mask_path=None):
    if tasks.empty or not np.all(tasks["run_hours"].to_numpy(dtype=int) == 1):
        raise ValueError("This compact script supports hourly aggregate tasks only; every run_hours must be 1.")
    return _build_one_hour_options(
        tasks,
        dc_cfg,
        ci_by_hour_loc,
        direct_wue_by_hour_loc,
        grid_water_by_hour_loc,
        pue_by_hour_loc,
        cloud_provider,
        include_transmission_carbon,
        delay_rounding,
        n_hours,
        option_workers=option_workers,
        option_chunk_size=option_chunk_size,
        policy_transmission_matrix_path=policy_transmission_matrix_path,
        policy_allowed_mask_path=policy_allowed_mask_path,
    )


def _benchmark_option_indices(tasks, options):
    indices = []
    for _, task in tasks.iterrows():
        task_id = int(task["task_id"])
        group = options.task_id_to_group[task_id]
        origin_idx = int(task["origin_idx"])
        arrival = int(task["arrival_hour"])
        local = group[options.option_dest[group] == origin_idx]
        if len(local) == 0:
            candidate = group
        else:
            immediate = local[options.option_start[local] == arrival]
            candidate = immediate if len(immediate) else local
        earliest = np.min(options.option_start[candidate])
        earliest_candidates = candidate[options.option_start[candidate] == earliest]
        indices.append(int(earliest_candidates[0]))
    return np.asarray(indices, dtype=np.int32)


def _normalization_scale(values, benchmark_idx, task_option_groups):
    scale = float(values[benchmark_idx].sum())
    if scale > 1e-12:
        return scale

    # Local/no-reallocation benchmarks usually have zero transmission cost.
    # Fall back to a feasible option-based scale so optional cost terms remain
    # dimensionless without forcing a near-zero denominator.
    fallback = 0.0
    for group in task_option_groups:
        if len(group):
            fallback += float(np.max(values[group]))
    if fallback > 1e-12:
        return fallback
    return 1.0


def _single_capacity_option_rows(options, n_locations):
    option_start = options.option_start.astype(np.int64, copy=False)
    option_finish = options.option_finish.astype(np.int64, copy=False)
    if not np.all(option_finish == option_start + 1):
        raise ValueError("Min-cost flow solver currently supports one-hour options only.")
    return option_start * int(n_locations) + options.option_dest.astype(np.int64, copy=False)


def _validate_fractional_solution(solution, tasks, options, dc_cfg, n_hours, it_capacity_kw):
    solution = np.asarray(solution, dtype=float)
    task_rows = _task_rows_for_options(tasks, options.option_task)
    task_sum = np.bincount(task_rows, weights=solution, minlength=len(tasks))
    max_task_error = float(np.max(np.abs(task_sum - 1.0))) if len(task_sum) else 0.0

    capacities = _capacity_matrix(dc_cfg, it_capacity_kw)
    task_resources = _task_resource_matrix(tasks)
    n_locations = len(dc_cfg)
    capacity_rows = _single_capacity_option_rows(options, n_locations).astype(np.int64, copy=False)
    usage = np.bincount(
        capacity_rows,
        weights=task_resources[task_rows, 0] * solution,
        minlength=int(n_hours) * n_locations,
    )
    b_ub = np.tile(capacities.reshape(-1), int(n_hours))
    max_capacity_violation = float(np.max(usage - b_ub)) if len(usage) else 0.0

    return max_task_error, max_capacity_violation


def _ortools_simple_min_cost_flow_class():
    try:
        from ortools.graph.python import min_cost_flow

        return min_cost_flow.SimpleMinCostFlow
    except ImportError:
        try:
            from ortools.graph import pywrapgraph

            return pywrapgraph.SimpleMinCostFlow
        except ImportError:
            return None


def _call_mcf_method(solver, snake_name, camel_name, *args):
    method = getattr(solver, snake_name, None)
    if method is None:
        method = getattr(solver, camel_name)
    return method(*args)


def _solve_scaled_min_cost_flow_ortools(n_tasks, n_capacity_rows, supply_int, capacity_int, option_indices, option_task_rows, option_capacity_rows, option_unit_cost_int):
    solver_class = _ortools_simple_min_cost_flow_class()
    if solver_class is None:
        raise ImportError("OR-Tools is not installed.")

    solver = solver_class()
    sink = int(n_tasks + n_capacity_rows)
    option_arcs = []

    for option_idx, task_row, capacity_row, unit_cost in zip(
        option_indices,
        option_task_rows,
        option_capacity_rows,
        option_unit_cost_int,
    ):
        task_row = int(task_row)
        capacity_row = int(capacity_row)
        arc = _call_mcf_method(
            solver,
            "add_arc_with_capacity_and_unit_cost",
            "AddArcWithCapacityAndUnitCost",
            task_row,
            int(n_tasks + capacity_row),
            int(supply_int[task_row]),
            int(unit_cost),
        )
        option_arcs.append((int(arc), int(option_idx), task_row))

    positive_capacity_rows = np.flatnonzero(capacity_int > 0)
    for capacity_row in positive_capacity_rows:
        _call_mcf_method(
            solver,
            "add_arc_with_capacity_and_unit_cost",
            "AddArcWithCapacityAndUnitCost",
            int(n_tasks + capacity_row),
            sink,
            int(capacity_int[capacity_row]),
            0,
        )

    for task_row in np.flatnonzero(supply_int > 0):
        _call_mcf_method(
            solver,
            "set_node_supply",
            "SetNodeSupply",
            int(task_row),
            int(supply_int[task_row]),
        )
    _call_mcf_method(
        solver,
        "set_node_supply",
        "SetNodeSupply",
        sink,
        -int(np.sum(supply_int, dtype=np.int64)),
    )

    status = _call_mcf_method(solver, "solve", "Solve")
    optimal_status = getattr(solver, "OPTIMAL", None)
    if optimal_status is None:
        optimal_status = getattr(solver_class, "OPTIMAL", None)
    if status != optimal_status:
        raise ValueError(f"OR-Tools min-cost flow failed with status={status}.")

    solution = np.zeros(len(option_indices), dtype=float)
    for local_idx, (arc, _, task_row) in enumerate(option_arcs):
        flow = _call_mcf_method(solver, "flow", "Flow", arc)
        if flow:
            solution[local_idx] = float(flow) / float(supply_int[task_row])
    return solution


def _solve_scaled_min_cost_flow_networkx(n_tasks, n_capacity_rows, supply_int, capacity_int, option_indices, option_task_rows, option_capacity_rows, option_unit_cost_int):
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("NetworkX is not installed.") from exc

    graph = nx.DiGraph()
    sink = int(n_tasks + n_capacity_rows)
    total_supply = int(np.sum(supply_int, dtype=np.int64))
    graph.add_node(sink, demand=total_supply)

    for task_row in np.flatnonzero(supply_int > 0):
        graph.add_node(int(task_row), demand=-int(supply_int[task_row]))

    for capacity_row in np.flatnonzero(capacity_int > 0):
        graph.add_edge(
            int(n_tasks + capacity_row),
            sink,
            capacity=int(capacity_int[capacity_row]),
            weight=0,
        )

    edge_to_option = {}
    for option_idx, task_row, capacity_row, unit_cost in zip(
        option_indices,
        option_task_rows,
        option_capacity_rows,
        option_unit_cost_int,
    ):
        task_row = int(task_row)
        capacity_row = int(capacity_row)
        if capacity_int[capacity_row] <= 0:
            continue
        task_node = task_row
        slot_node = int(n_tasks + capacity_row)
        key = (task_node, slot_node)
        unit_cost = int(unit_cost)
        previous = edge_to_option.get(key)
        if previous is None or unit_cost < previous[1]:
            graph.add_edge(
                task_node,
                slot_node,
                capacity=int(supply_int[task_row]),
                weight=unit_cost,
            )
            edge_to_option[key] = (int(option_idx), unit_cost)

    _, flow_dict = nx.network_simplex(graph)

    solution_by_option = np.zeros(len(option_indices), dtype=float)
    option_to_local = {int(option_idx): local_idx for local_idx, option_idx in enumerate(option_indices)}
    for (task_node, slot_node), (option_idx, _) in edge_to_option.items():
        flow = flow_dict.get(task_node, {}).get(slot_node, 0)
        if flow:
            solution_by_option[option_to_local[option_idx]] = float(flow) / float(supply_int[task_node])
    return solution_by_option


def solve_single_capacity_min_cost_flow(tasks, options, dc_cfg, n_hours, objective_values=None, it_capacity_kw=None, backend="auto", resource_scale=1_000_000.0, cost_scale=1_000_000.0, feasibility_tolerance=1e-7):
    if not np.isfinite(resource_scale) or float(resource_scale) <= 0.0:
        raise ValueError("--mcf-resource-scale must be positive.")
    if not np.isfinite(cost_scale) or float(cost_scale) <= 0.0:
        raise ValueError("--mcf-cost-scale must be positive.")

    objective_values = options.option_objective if objective_values is None else np.asarray(objective_values, dtype=float)
    resources = _task_resource_matrix(tasks).reshape(-1)
    capacities = _capacity_matrix(dc_cfg, it_capacity_kw).reshape(-1)
    n_tasks = len(tasks)
    n_locations = len(dc_cfg)
    n_capacity_rows = int(n_hours) * n_locations
    task_rows_by_option = _task_rows_for_options(tasks, options.option_task)
    capacity_rows_by_option = _single_capacity_option_rows(options, n_locations)

    solution = np.zeros(len(options.option_task), dtype=float)
    task_ids = tasks["task_id"].to_numpy(dtype=int)
    positive_task_mask = resources > 1e-12
    zero_task_rows = np.flatnonzero(~positive_task_mask)
    for task_row in zero_task_rows:
        group = options.task_id_to_group[int(task_ids[task_row])]
        best_local = int(np.argmin(objective_values[group]))
        solution[int(group[best_local])] = 1.0

    supply_int = np.zeros(n_tasks, dtype=np.int64)
    supply_int[positive_task_mask] = np.maximum(
        1,
        np.ceil(resources[positive_task_mask] * float(resource_scale)).astype(np.int64),
    )
    capacity_int = np.floor(
        np.tile(capacities, int(n_hours)) * float(resource_scale)
    ).astype(np.int64)
    capacity_int = np.maximum(capacity_int, 0)

    total_supply = int(np.sum(supply_int, dtype=np.int64))
    total_capacity = int(np.sum(capacity_int, dtype=np.int64))
    if total_capacity < total_supply:
        raise ValueError(
            "Scaled min-cost flow is infeasible before solving: "
            f"scaled_supply={total_supply}, scaled_capacity={total_capacity}. "
            "Increase --mcf-resource-scale if this is caused by rounding."
        )
    if total_supply == 0:
        return solution

    positive_option_mask = positive_task_mask[task_rows_by_option]
    positive_option_mask &= capacity_int[capacity_rows_by_option] > 0
    positive_option_indices = np.flatnonzero(positive_option_mask)
    if len(positive_option_indices) == 0:
        raise ValueError("No positive-capacity min-cost-flow arcs were created.")

    option_task_rows = task_rows_by_option[positive_option_indices].astype(np.int64, copy=False)
    option_capacity_rows = capacity_rows_by_option[positive_option_indices].astype(np.int64, copy=False)
    edge_counts = np.bincount(option_task_rows, minlength=n_tasks)
    missing_tasks = np.flatnonzero((supply_int > 0) & (edge_counts == 0))
    if len(missing_tasks):
        preview = ", ".join(str(int(task_ids[idx])) for idx in missing_tasks[:10])
        raise ValueError(
            "Some positive-resource tasks have no scaled feasible MCF option. "
            f"task_id preview: {preview}"
        )

    unit_cost = objective_values[positive_option_indices] / resources[option_task_rows]
    option_unit_cost_int = np.rint(unit_cost * float(cost_scale)).astype(np.int64)

    selected_backend = str(backend).lower()
    if selected_backend not in ("auto", "ortools", "networkx"):
        raise ValueError("--mcf-backend must be auto, ortools, or networkx.")
    if selected_backend == "auto":
        selected_backend = "ortools" if _ortools_simple_min_cost_flow_class() is not None else "networkx"

    solve_start = time.perf_counter()
    _progress(
        f"Running min-cost flow: backend={selected_backend}, "
        f"tasks={int(np.sum(supply_int > 0))}, slots={n_capacity_rows}, "
        f"arcs={len(positive_option_indices)}, resource_scale={resource_scale:g}, "
        f"cost_scale={cost_scale:g}"
    )
    if selected_backend == "ortools":
        local_solution = _solve_scaled_min_cost_flow_ortools(
            n_tasks,
            n_capacity_rows,
            supply_int,
            capacity_int,
            positive_option_indices,
            option_task_rows,
            option_capacity_rows,
            option_unit_cost_int,
        )
    else:
        local_solution = _solve_scaled_min_cost_flow_networkx(
            n_tasks,
            n_capacity_rows,
            supply_int,
            capacity_int,
            positive_option_indices,
            option_task_rows,
            option_capacity_rows,
            option_unit_cost_int,
        )

    solution[positive_option_indices] = local_solution
    max_task_error, max_capacity_violation = _validate_fractional_solution(
        solution,
        tasks,
        options,
        dc_cfg,
        n_hours,
        it_capacity_kw,
    )
    objective = float(np.dot(objective_values, solution))
    _progress(
        f"min-cost flow finished: original_objective={objective:.12g}, "
        f"max_task_error={max_task_error:.3g}, "
        f"max_capacity_violation={max_capacity_violation:.3g}",
        solve_start,
    )
    if max_task_error > max(float(feasibility_tolerance), 1e-9):
        raise ValueError(f"MCF solution violates task assignment equalities by {max_task_error:.6g}.")
    if max_capacity_violation > max(float(feasibility_tolerance), 1e-9):
        raise ValueError(
            f"MCF solution violates original capacity constraints by {max_capacity_violation:.6g}. "
            "Increase --mcf-resource-scale or relax --mcf-feasibility-tolerance."
        )
    return solution


def _ordered_tasks_for_greedy(tasks, options, order_mode):
    if order_mode == "arrival":
        return tasks.sort_values(["arrival_hour", "latest_start_hour", "task_type", "task_id"])
    if order_mode == "constrained":
        constrained_first = tasks.copy()
        constrained_first["_option_count"] = [
            len(options.task_id_to_group[int(task_id)])
            for task_id in constrained_first["task_id"].to_numpy(dtype=int)
        ]
        return constrained_first.sort_values(
            ["_option_count", "latest_start_hour", "arrival_hour", "task_type", "task_id"]
        )
    raise ValueError(f"Unknown greedy order_mode: {order_mode}")


def _greedy_fractional(tasks, options, dc_cfg, n_hours, objective_values=None, tolerance=1e-9, order_mode="arrival", it_capacity_kw=None, progress_label=None):
    progress_start = time.perf_counter()
    objective_values = options.option_objective if objective_values is None else np.asarray(objective_values, dtype=float)
    capacities = _capacity_matrix(dc_cfg, it_capacity_kw)
    usage = np.zeros((n_hours, len(dc_cfg), capacities.shape[1]), dtype=float)
    choice = np.zeros(len(options.option_task), dtype=float)
    tasks_sorted = _ordered_tasks_for_greedy(tasks, options, order_mode)
    total_tasks = len(tasks_sorted)
    progress_step = max(1, total_tasks // 20)
    if progress_label:
        _progress(f"{progress_label}: 0/{total_tasks} tasks")

    for task_count, (_, task) in enumerate(tasks_sorted.iterrows(), start=1):
        group = options.task_id_to_group[int(task["task_id"])]
        resources = _task_capacity_vector(task)
        remaining = 1.0
        for option_idx in group[np.argsort(objective_values[group])]:
            dest_idx = int(options.option_dest[option_idx])
            start = int(options.option_start[option_idx])
            finish = int(options.option_finish[option_idx])
            resource_limits = []
            for res_idx, coeff in enumerate(resources):
                if coeff <= tolerance:
                    continue
                slack = capacities[dest_idx, res_idx] - usage[start:finish, dest_idx, res_idx]
                resource_limits.append(float(np.min(slack) / coeff))
            max_fraction = 1.0 if not resource_limits else min(resource_limits)
            take = min(remaining, max(0.0, max_fraction))
            if take <= tolerance:
                continue
            usage[start:finish, dest_idx, :] += resources * take
            choice[int(option_idx)] += take
            remaining -= take
            if remaining <= tolerance:
                break
        if remaining > tolerance:
            raise ValueError(
                f"Fractional greedy could not place task_id={task['task_id']} "
                f"(remaining fraction={remaining:.6g})."
            )
        if progress_label and (task_count % progress_step == 0 or task_count == total_tasks):
            pct = 100.0 * task_count / max(total_tasks, 1)
            _progress(f"{progress_label}: {task_count}/{total_tasks} tasks ({pct:.1f}%)", progress_start)
    return choice


def _sort_options_by_start(options, candidates):
    candidates = np.asarray(candidates, dtype=np.int32)
    if len(candidates) <= 1:
        return candidates
    return candidates[np.argsort(options.option_start[candidates], kind="mergesort")]


def _sort_options_by_objective_then_start(options, candidates, objective_values):
    candidates = np.asarray(candidates, dtype=np.int32)
    if len(candidates) <= 1:
        return candidates
    order = np.lexsort((options.option_start[candidates], objective_values[candidates]))
    return candidates[order]


def _sort_options_by_start_then_objective(options, candidates, objective_values):
    candidates = np.asarray(candidates, dtype=np.int32)
    if len(candidates) <= 1:
        return candidates
    order = np.lexsort((objective_values[candidates], options.option_start[candidates]))
    return candidates[order]


def _local_immediate_training_overflow_candidates(task, options):
    task_id = int(task["task_id"])
    group = options.task_id_to_group[task_id]
    origin_idx = int(task["origin_idx"])
    arrival = int(task["arrival_hour"])
    local = group[options.option_dest[group].astype(np.int32, copy=False) == origin_idx]
    immediate = local[options.option_start[local].astype(np.int32, copy=False) == arrival]
    immediate = _sort_options_by_start(options, immediate)
    if str(task["task_type"]) != "training":
        return immediate
    delayed = local[options.option_start[local].astype(np.int32, copy=False) > arrival]
    delayed = _sort_options_by_start(options, delayed)
    return np.concatenate((immediate, delayed)).astype(np.int32, copy=False)


def _immediate_overflow_candidates(task, options, objective_values):
    task_id = int(task["task_id"])
    group = options.task_id_to_group[task_id]
    arrival = int(task["arrival_hour"])
    starts = options.option_start[group].astype(np.int32, copy=False)
    immediate = group[starts == arrival]
    immediate = _sort_options_by_objective_then_start(options, immediate, objective_values)

    delayed = group[starts > arrival]
    delayed = _sort_options_by_start_then_objective(options, delayed, objective_values)
    if len(delayed) == 0:
        return immediate
    return np.concatenate((immediate, delayed)).astype(np.int32, copy=False)


def _local_training_time_shift_candidates(task, options, objective_values):
    task_id = int(task["task_id"])
    group = options.task_id_to_group[task_id]
    origin_idx = int(task["origin_idx"])
    arrival = int(task["arrival_hour"])
    local = group[options.option_dest[group].astype(np.int32, copy=False) == origin_idx]
    if str(task["task_type"]) != "training":
        local = local[options.option_start[local].astype(np.int32, copy=False) == arrival]
    return _sort_options_by_objective_then_start(options, local, objective_values)


def _greedy_fractional_from_candidate_builder(tasks, options, dc_cfg, n_hours, candidate_builder, tolerance=1e-9, it_capacity_kw=None, progress_label=None):
    progress_start = time.perf_counter()
    capacities = _capacity_matrix(dc_cfg, it_capacity_kw)
    usage = np.zeros((n_hours, len(dc_cfg), capacities.shape[1]), dtype=float)
    choice = np.zeros(len(options.option_task), dtype=float)

    constrained_first = tasks.copy()
    constrained_first["_option_count"] = [
        len(candidate_builder(task))
        for _, task in constrained_first.iterrows()
    ]
    tasks_sorted = constrained_first.sort_values(
        ["_option_count", "latest_start_hour", "arrival_hour", "task_type", "task_id"]
    )

    total_tasks = len(tasks_sorted)
    progress_step = max(1, total_tasks // 20)
    if progress_label:
        _progress(f"{progress_label}: 0/{total_tasks} tasks")

    for task_count, (_, task) in enumerate(tasks_sorted.iterrows(), start=1):
        candidates = candidate_builder(task)
        if len(candidates) == 0:
            raise ValueError(f"No legal greedy candidate for task_id={task['task_id']}.")
        resources = _task_capacity_vector(task)
        remaining = 1.0
        for option_idx in candidates:
            dest_idx = int(options.option_dest[option_idx])
            start = int(options.option_start[option_idx])
            finish = int(options.option_finish[option_idx])
            resource_limits = []
            for res_idx, coeff in enumerate(resources):
                if coeff <= tolerance:
                    continue
                slack = capacities[dest_idx, res_idx] - usage[start:finish, dest_idx, res_idx]
                resource_limits.append(float(np.min(slack) / coeff))
            max_fraction = 1.0 if not resource_limits else min(resource_limits)
            take = min(remaining, max(0.0, max_fraction))
            if take <= tolerance:
                continue
            usage[start:finish, dest_idx, :] += resources * take
            choice[int(option_idx)] += take
            remaining -= take
            if remaining <= tolerance:
                break
        if remaining > tolerance:
            raise ValueError(
                f"Baseline greedy could not place task_id={task['task_id']} "
                f"(remaining fraction={remaining:.6g})."
            )
        if progress_label and (task_count % progress_step == 0 or task_count == total_tasks):
            pct = 100.0 * task_count / max(total_tasks, 1)
            _progress(f"{progress_label}: {task_count}/{total_tasks} tasks ({pct:.1f}%)", progress_start)
    return choice


def solve_local_immediate_training_overflow_greedy(tasks, options, dc_cfg, n_hours, it_capacity_kw=None, progress_label=None):
    return _greedy_fractional_from_candidate_builder(
        tasks,
        options,
        dc_cfg,
        n_hours,
        lambda task: _local_immediate_training_overflow_candidates(task, options),
        it_capacity_kw=it_capacity_kw,
        progress_label=progress_label,
    )


def solve_local_training_time_shift_greedy(tasks, options, dc_cfg, n_hours, objective_values=None, it_capacity_kw=None, progress_label=None):
    objective_values = options.option_objective if objective_values is None else np.asarray(objective_values, dtype=float)
    return _greedy_fractional_from_candidate_builder(
        tasks,
        options,
        dc_cfg,
        n_hours,
        lambda task: _local_training_time_shift_candidates(task, options, objective_values),
        it_capacity_kw=it_capacity_kw,
        progress_label=progress_label,
    )


def solve_immediate_overflow_greedy(tasks, options, dc_cfg, n_hours, objective_values=None, it_capacity_kw=None, progress_label=None):
    objective_values = options.option_objective if objective_values is None else np.asarray(objective_values, dtype=float)
    return _greedy_fractional_from_candidate_builder(
        tasks,
        options,
        dc_cfg,
        n_hours,
        lambda task: _immediate_overflow_candidates(task, options, objective_values),
        it_capacity_kw=it_capacity_kw,
        progress_label=progress_label,
    )


def _objective_parts(options, include_transmission_cost, include_transmission_carbon):
    carbon = options.option_compute_carbon_kg.copy()
    if include_transmission_carbon:
        carbon = carbon + options.option_transmission_carbon_kg
    water = options.option_water_m3.copy()
    tx_cost = options.option_tx_cost_usd if include_transmission_cost else np.zeros_like(carbon)
    return carbon, water, tx_cost


def _objective_scales(tasks, options, parts, objective_normalization):
    if objective_normalization == "none":
        return (1.0, 1.0, 1.0)
    if objective_normalization != "benchmark":
        raise ValueError("--objective-normalization must be benchmark or none.")
    benchmark_idx = _benchmark_option_indices(tasks, options)
    return tuple(_normalization_scale(part, benchmark_idx, options.task_option_groups) for part in parts)


def _combine_objective_parts(parts, scales, carbon_weight, water_weight, tx_weight):
    carbon, water, tx_cost = parts
    carbon_scale, water_scale, tx_cost_scale = scales
    return carbon_weight * carbon / carbon_scale + water_weight * water / water_scale + tx_weight * tx_cost / tx_cost_scale


def build_objective_values(tasks, options, carbon_weight, water_weight, tx_weight, include_transmission_cost, include_transmission_carbon, objective_normalization, objective_scales=None):
    parts = _objective_parts(options, include_transmission_cost, include_transmission_carbon)
    scales = (
        _objective_scales(tasks, options, parts, objective_normalization)
        if objective_scales is None
        else tuple(float(value) for value in objective_scales)
    )
    return _combine_objective_parts(parts, scales, carbon_weight, water_weight, tx_weight)


def build_annual_average_objective_scores(tasks, options, ci_by_hour_loc, carbon_weight, water_weight, tx_weight, include_transmission_cost, include_transmission_carbon, objective_normalization, objective_scales=None):
    actual_parts = _objective_parts(options, include_transmission_cost, include_transmission_carbon)
    scales = (
        _objective_scales(tasks, options, actual_parts, objective_normalization)
        if objective_scales is None
        else tuple(float(value) for value in objective_scales)
    )
    annual_ci = np.mean(ci_by_hour_loc, axis=0)
    annual_carbon = options.option_facility_energy_kwh * annual_ci[options.option_dest] / 1000.0
    if include_transmission_carbon:
        annual_carbon = annual_carbon + options.option_transmission_carbon_kg
    annual_parts = (annual_carbon, actual_parts[1], actual_parts[2])
    return _combine_objective_parts(annual_parts, scales, carbon_weight, water_weight, tx_weight)


def build_baseline_specs(tasks, options, ci_by_hour_loc, carbon_weight, water_weight, tx_weight, include_transmission_cost, include_transmission_carbon, objective_normalization, objective_scales=None):
    hourly_objective = np.asarray(options.option_objective, dtype=float)
    annual_objective = build_annual_average_objective_scores(
        tasks,
        options,
        ci_by_hour_loc,
        carbon_weight,
        water_weight,
        tx_weight,
        include_transmission_cost,
        include_transmission_carbon,
        objective_normalization,
        objective_scales=objective_scales,
    )
    return [
        (
            "local_immediate_training_overflow_constrained",
            hourly_objective,
            "local_immediate_training_overflow_greedy",
        ),
        (
            "local_training_time_shift_greedy_constrained",
            hourly_objective,
            "local_training_time_shift_greedy",
        ),
        (
            "annual_average_immediate_overflow_greedy_constrained",
            annual_objective,
            "immediate_overflow_greedy",
        ),
        (
            "hourly_immediate_overflow_greedy_constrained",
            hourly_objective,
            "immediate_overflow_greedy",
        ),
        ("hourly_arrival_greedy", hourly_objective, "arrival_greedy"),
    ]


def materialize_assignment(solution, tasks, options, dc_cfg, objective_values=None, tolerance=1e-8):
    objective_values = options.option_objective if objective_values is None else np.asarray(objective_values, dtype=float)
    selected = np.flatnonzero(solution > tolerance)
    if len(selected) == 0:
        return _empty_assignment_frame()

    task_rows = _task_rows_for_options(tasks, options.option_task[selected])
    locations = np.asarray([dc["location"] for dc in dc_cfg], dtype=object)
    fraction = np.asarray(solution[selected], dtype=float)
    start = options.option_start[selected].astype(int, copy=False)
    finish = options.option_finish[selected].astype(int, copy=False)
    dest_idx = options.option_dest[selected].astype(int, copy=False)
    arrival = tasks["arrival_hour"].to_numpy(dtype=int)[task_rows]
    gpu_mem_req = (
        tasks["gpu_mem_req"].to_numpy(dtype=float)
        if "gpu_mem_req" in tasks.columns
        else np.zeros(len(tasks), dtype=float)
    )

    data = {
        "task_id": tasks["task_id"].to_numpy(dtype=int)[task_rows],
        "task_type": tasks["task_type"].to_numpy(dtype=object)[task_rows],
        "arrival_hour": arrival,
        "start_hour": start,
        "finish_hour": finish,
        "delay_hours": start - arrival,
        "origin": tasks["origin"].to_numpy(dtype=object)[task_rows],
        "location": locations[dest_idx],
        "duration_minutes": tasks["duration_minutes"].to_numpy(dtype=float)[task_rows],
        "run_hours": tasks["run_hours"].to_numpy(dtype=int)[task_rows],
        "cores_req": tasks["cores_req"].to_numpy(dtype=float)[task_rows],
        "gpu_req": tasks["gpu_req"].to_numpy(dtype=float)[task_rows],
        "gpu_mem_req": gpu_mem_req[task_rows],
        "mem_req": tasks["mem_req"].to_numpy(dtype=float)[task_rows],
        "bandwidth_gb": tasks["bandwidth_gb"].to_numpy(dtype=float)[task_rows],
        "source_task_count": tasks["source_task_count"].to_numpy(dtype=float)[task_rows],
        "fraction": fraction,
        "pue": options.option_pue[selected],
        "compute_energy_kwh_proxy": options.option_compute_energy_kwh[selected] * fraction,
        "facility_energy_kwh_pue": options.option_facility_energy_kwh[selected] * fraction,
        "compute_carbon_kg": options.option_compute_carbon_kg[selected] * fraction,
        "direct_water_m3": options.option_direct_water_m3[selected] * fraction,
        "grid_water_m3": options.option_grid_water_m3[selected] * fraction,
        "water_m3": options.option_water_m3[selected] * fraction,
        "transmission_energy_kwh": options.option_transmission_energy_kwh[selected] * fraction,
        "transmission_carbon_kg": options.option_transmission_carbon_kg[selected] * fraction,
        "carbon_kg": options.option_carbon_kg[selected] * fraction,
        "tx_cost": options.option_tx_cost_usd[selected] * fraction,
        "objective": objective_values[selected] * fraction,
    }
    return pd.DataFrame(data)


def _safe_scaled_ratio(numerator, denominator, scale=1.0):
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / denominator * float(scale)


def _effective_horizon_hours(n_hours, assignment):
    if n_hours is not None:
        return max(int(n_hours), 1)
    if assignment.empty or "finish_hour" not in assignment.columns:
        return 1
    return max(int(assignment["finish_hour"].max()), 1)


def _assignment_with_power_breakdown(assignment):
    rows = assignment.copy()
    if rows.empty:
        for col in [
            "_compute_power_kw",
            "_facility_power_kw",
            "_cpu_energy_kwh",
            "_gpu_energy_kwh",
            "_mem_energy_kwh",
        ]:
            rows[col] = pd.Series(dtype=float)
        return rows

    fraction = rows["fraction"].astype(float)
    run_hours = np.maximum(rows["run_hours"].astype(float), 1e-12)
    rows["_compute_power_kw"] = _it_power_kw(
        rows["cores_req"].astype(float),
        rows["gpu_req"].astype(float),
        rows["mem_req"].astype(float),
    ) * fraction
    rows["_facility_power_kw"] = rows["facility_energy_kwh_pue"].astype(float) / run_hours
    rows["_cpu_energy_kwh"] = (
        rows["cores_req"].astype(float)
        * CPU_KW_PER_CORE
        * rows["run_hours"].astype(float)
        * fraction
    )
    rows["_gpu_energy_kwh"] = (
        rows["gpu_req"].astype(float)
        * GPU_KW_PER_UNIT
        * rows["run_hours"].astype(float)
        * fraction
    )
    rows["_mem_energy_kwh"] = (
        rows["mem_req"].astype(float)
        * MEM_KW_PER_GB
        * rows["run_hours"].astype(float)
        * fraction
    )
    return rows


def _peak_grouped_by_start_hour(rows, value_col):
    if rows.empty:
        return 0.0
    grouped = rows.groupby("start_hour", sort=False)[value_col].sum()
    if grouped.empty:
        return 0.0
    return float(grouped.max())


def _task_type_metrics(rows, n_hours, prefix):
    selected = rows.loc[rows["task_type"] == prefix] if not rows.empty else rows
    compute_energy = float(selected["compute_energy_kwh_proxy"].sum()) if not selected.empty else 0.0
    facility_energy = float(selected["facility_energy_kwh_pue"].sum()) if not selected.empty else 0.0
    carbon = float(selected["compute_carbon_kg"].sum()) if not selected.empty else 0.0
    return {
        f"{prefix}_compute_energy_kwh_proxy": compute_energy,
        f"{prefix}_facility_energy_kwh_pue": facility_energy,
        f"{prefix}_compute_carbon_kg": carbon,
        f"{prefix}_mean_compute_power_kw": compute_energy / max(int(n_hours), 1),
        f"{prefix}_peak_compute_power_kw": _peak_grouped_by_start_hour(selected, "_compute_power_kw"),
        f"{prefix}_mean_facility_power_kw": facility_energy / max(int(n_hours), 1),
        f"{prefix}_peak_facility_power_kw": _peak_grouped_by_start_hour(selected, "_facility_power_kw"),
    }


def summarize_assignment(name, assignment, runtime_seconds=None, n_hours=None):
    runtime_seconds = 0.0 if runtime_seconds is None else float(runtime_seconds)
    horizon_hours = _effective_horizon_hours(n_hours, assignment)
    if assignment.empty:
        base = {
            "strategy": name,
            "runtime_seconds": runtime_seconds,
            "task_fraction_served": 0.0,
            "unique_tasks_touched": 0,
        }
        for key in [
            "compute_energy_kwh_proxy", "facility_energy_kwh_pue", "compute_carbon_kg",
            "direct_water_m3", "grid_water_m3", "water_m3", "transmission_energy_kwh",
            "transmission_carbon_kg", "carbon_kg_objective_scope", "carbon_kg_compute_plus_tx",
            "tx_cost_usd", "objective", "load_weighted_delay_hours", "max_delay_hours",
            "cpu_energy_kwh", "gpu_energy_kwh", "mem_energy_kwh",
        ]:
            base[key] = 0.0
        base.update(_task_type_metrics(_assignment_with_power_breakdown(assignment), horizon_hours, "training"))
        base.update(_task_type_metrics(_assignment_with_power_breakdown(assignment), horizon_hours, "inference"))
        return base

    fraction = assignment["fraction"].to_numpy(dtype=float)
    delay = assignment["delay_hours"].to_numpy(dtype=float)
    weighted_delay = float(np.sum(delay * fraction) / max(np.sum(fraction), 1e-12))
    rows = _assignment_with_power_breakdown(assignment)
    summary = {
        "strategy": name,
        "runtime_seconds": runtime_seconds,
        "task_fraction_served": float(assignment["fraction"].sum()),
        "unique_tasks_touched": int(assignment["task_id"].nunique()),
        "compute_energy_kwh_proxy": float(assignment["compute_energy_kwh_proxy"].sum()),
        "facility_energy_kwh_pue": float(assignment["facility_energy_kwh_pue"].sum()),
        "compute_carbon_kg": float(assignment["compute_carbon_kg"].sum()),
        "direct_water_m3": float(assignment["direct_water_m3"].sum()),
        "grid_water_m3": float(assignment["grid_water_m3"].sum()),
        "water_m3": float(assignment["water_m3"].sum()),
        "transmission_energy_kwh": float(assignment["transmission_energy_kwh"].sum()),
        "transmission_carbon_kg": float(assignment["transmission_carbon_kg"].sum()),
        "carbon_kg_objective_scope": float(assignment["carbon_kg"].sum()),
        "carbon_kg_compute_plus_tx": float(assignment["compute_carbon_kg"].sum() + assignment["transmission_carbon_kg"].sum()),
        "tx_cost_usd": float(assignment["tx_cost"].sum()),
        "objective": float(assignment["objective"].sum()),
        "load_weighted_delay_hours": weighted_delay,
        "max_delay_hours": float(assignment["delay_hours"].max()),
        "cpu_energy_kwh": float(rows["_cpu_energy_kwh"].sum()),
        "gpu_energy_kwh": float(rows["_gpu_energy_kwh"].sum()),
        "mem_energy_kwh": float(rows["_mem_energy_kwh"].sum()),
    }
    summary.update(_task_type_metrics(rows, horizon_hours, "training"))
    summary.update(_task_type_metrics(rows, horizon_hours, "inference"))
    return summary


def order_summary_columns(summary_df):
    return summary_df.reindex(columns=SUMMARY_COLUMNS)


def _with_output_column_labels(df, output_columns):
    return df.rename(columns={col: output_columns[col] for col in df.columns if col in output_columns})


def _task_type_ratio_frame(records, locations, location_col, weight_col, prefix, total_denominator=None):
    columns = [
        "location",
        f"{prefix}_training_within_location",
        f"{prefix}_inference_within_location",
        f"{prefix}_training_of_all_tasks",
        f"{prefix}_inference_of_all_tasks",
    ]
    if records.empty:
        return pd.DataFrame(
            [
                {
                    "location": location,
                    f"{prefix}_training_within_location": 0.0,
                    f"{prefix}_inference_within_location": 0.0,
                    f"{prefix}_training_of_all_tasks": 0.0,
                    f"{prefix}_inference_of_all_tasks": 0.0,
                }
                for location in locations
            ],
            columns=columns,
        )

    grouped = (
        records.groupby([location_col, "task_type"], as_index=False)[weight_col]
        .sum()
        .pivot(index=location_col, columns="task_type", values=weight_col)
        .reindex(locations)
        .fillna(0.0)
    )
    training = grouped["training"] if "training" in grouped.columns else pd.Series(0.0, index=grouped.index)
    inference = grouped["inference"] if "inference" in grouped.columns else pd.Series(0.0, index=grouped.index)
    location_total = training + inference
    all_total = float(location_total.sum()) if total_denominator is None else float(total_denominator)
    out = pd.DataFrame({"location": grouped.index})
    location_total_values = location_total.to_numpy(dtype=float)
    training_values = training.to_numpy(dtype=float)
    inference_values = inference.to_numpy(dtype=float)
    out[f"{prefix}_training_within_location"] = np.divide(
        training_values,
        location_total_values,
        out=np.zeros(len(out), dtype=float),
        where=location_total_values > 1e-12,
    )
    out[f"{prefix}_inference_within_location"] = np.divide(
        inference_values,
        location_total_values,
        out=np.zeros(len(out), dtype=float),
        where=location_total_values > 1e-12,
    )
    if all_total > 1e-12:
        out[f"{prefix}_training_of_all_tasks"] = training_values / all_total
        out[f"{prefix}_inference_of_all_tasks"] = inference_values / all_total
    else:
        out[f"{prefix}_training_of_all_tasks"] = 0.0
        out[f"{prefix}_inference_of_all_tasks"] = 0.0
    return out.reindex(columns=columns)


def summarize_task_type_by_location(name, tasks, assignment, dc_cfg):
    locations = [dc["location"] for dc in dc_cfg]
    source = tasks[["origin", "task_type", "source_task_count"]].copy()
    source["_task_fraction"] = source["source_task_count"].astype(float)
    total_source_tasks = float(source["_task_fraction"].sum())
    assigned = assignment.copy()
    if not assigned.empty:
        assigned["_task_fraction"] = (
            assigned["source_task_count"].astype(float)
            * assigned["fraction"].astype(float)
        )
    before = _task_type_ratio_frame(
        source,
        locations,
        location_col="origin",
        weight_col="_task_fraction",
        prefix="before",
        total_denominator=total_source_tasks,
    )
    after = _task_type_ratio_frame(
        assigned,
        locations,
        location_col="location",
        weight_col="_task_fraction",
        prefix="after",
        total_denominator=total_source_tasks,
    )
    out = before.merge(after, on="location", how="outer").fillna(0.0)
    out.insert(0, "strategy", name)
    return out


def _peak_value_and_hour(hourly_values):
    if len(hourly_values) == 0:
        return 0.0, 0
    peak_hour = int(np.argmax(hourly_values))
    return float(hourly_values[peak_hour]), peak_hour


def _mean_daily_peak_valley(hourly_values):
    if len(hourly_values) == 0:
        return 0.0
    ranges = []
    for day_start in range(0, len(hourly_values), 24):
        day = hourly_values[day_start:day_start + 24]
        if len(day) > 0:
            ranges.append(float(np.max(day) - np.min(day)))
    return float(np.mean(ranges)) if ranges else 0.0


def _max_hourly_ramp(hourly_values):
    if len(hourly_values) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(hourly_values))))


def _add_hourly_values(hourly_values, rows, value_col, n_hours):
    if rows.empty:
        return
    hours = rows["start_hour"].astype(int).to_numpy()
    values = rows[value_col].astype(float).to_numpy()
    valid = (hours >= 0) & (hours < int(n_hours))
    if np.any(valid):
        np.add.at(hourly_values, hours[valid], values[valid])


def _country_task_type_metrics(rows, n_hours, prefix):
    selected = rows.loc[rows["task_type"] == prefix] if not rows.empty else rows
    compute_energy = float(selected["compute_energy_kwh_proxy"].sum()) if not selected.empty else 0.0
    facility_energy = float(selected["facility_energy_kwh_pue"].sum()) if not selected.empty else 0.0
    carbon = float(selected["compute_carbon_kg"].sum()) if not selected.empty else 0.0

    hourly_compute = np.zeros(int(n_hours), dtype=float)
    hourly_facility = np.zeros(int(n_hours), dtype=float)
    _add_hourly_values(hourly_compute, selected, "_compute_power_kw", n_hours)
    _add_hourly_values(hourly_facility, selected, "_facility_power_kw", n_hours)

    return {
        f"{prefix}_compute_energy_kwh_proxy": compute_energy,
        f"{prefix}_facility_energy_kwh_pue": facility_energy,
        f"{prefix}_compute_carbon_kg": carbon,
        f"{prefix}_mean_compute_power_kw": compute_energy / max(int(n_hours), 1),
        f"{prefix}_peak_compute_power_kw": float(np.max(hourly_compute)) if len(hourly_compute) else 0.0,
        f"{prefix}_mean_facility_power_kw": facility_energy / max(int(n_hours), 1),
        f"{prefix}_peak_facility_power_kw": float(np.max(hourly_facility)) if len(hourly_facility) else 0.0,
    }


def build_country_summary_frame(name, assignment, dc_cfg, n_hours, ci_by_hour_loc, high_carbon_quantile):
    n_hours = max(int(n_hours), 1)
    locations = [dc["location"] for dc in dc_cfg]
    rows = _assignment_with_power_breakdown(assignment)
    total_facility_energy = float(rows["facility_energy_kwh_pue"].sum()) if not rows.empty else 0.0
    total_compute_carbon = float(rows["compute_carbon_kg"].sum()) if not rows.empty else 0.0
    ci = np.asarray(ci_by_hour_loc, dtype=float)[:n_hours, :]

    records = []
    for loc_idx, location in enumerate(locations):
        loc_rows = rows.loc[rows["location"] == location] if not rows.empty else rows
        hourly_compute = np.zeros(n_hours, dtype=float)
        hourly_facility = np.zeros(n_hours, dtype=float)
        _add_hourly_values(hourly_compute, loc_rows, "_compute_power_kw", n_hours)
        _add_hourly_values(hourly_facility, loc_rows, "_facility_power_kw", n_hours)

        compute_energy = float(loc_rows["compute_energy_kwh_proxy"].sum()) if not loc_rows.empty else 0.0
        facility_energy = float(loc_rows["facility_energy_kwh_pue"].sum()) if not loc_rows.empty else 0.0
        compute_carbon = float(loc_rows["compute_carbon_kg"].sum()) if not loc_rows.empty else 0.0
        peak_compute_power, peak_compute_hour = _peak_value_and_hour(hourly_compute)
        peak_facility_power, peak_facility_hour = _peak_value_and_hour(hourly_facility)

        loc_ci = ci[:, loc_idx] if ci.ndim == 2 and loc_idx < ci.shape[1] else np.asarray([], dtype=float)
        finite_ci = loc_ci[np.isfinite(loc_ci)]
        if len(finite_ci) > 0:
            high_carbon_threshold = float(np.quantile(finite_ci, float(high_carbon_quantile)))
            high_carbon_mask = np.isfinite(loc_ci) & (loc_ci >= high_carbon_threshold)
            high_carbon_energy = float(np.sum(hourly_facility[high_carbon_mask]))
        else:
            high_carbon_energy = 0.0

        record = {
            "strategy": name,
            "location": location,
            "compute_energy_kwh_proxy": compute_energy,
            "facility_energy_kwh_pue": facility_energy,
            "compute_carbon_kg": compute_carbon,
            "facility_energy_share_pct": _safe_scaled_ratio(facility_energy, total_facility_energy, 100.0),
            "compute_carbon_share_pct": _safe_scaled_ratio(compute_carbon, total_compute_carbon, 100.0),
            "cpu_energy_kwh": float(loc_rows["_cpu_energy_kwh"].sum()) if not loc_rows.empty else 0.0,
            "gpu_energy_kwh": float(loc_rows["_gpu_energy_kwh"].sum()) if not loc_rows.empty else 0.0,
            "mem_energy_kwh": float(loc_rows["_mem_energy_kwh"].sum()) if not loc_rows.empty else 0.0,
            "mean_compute_power_kw": compute_energy / n_hours,
            "peak_compute_power_kw": peak_compute_power,
            "peak_compute_power_hour": peak_compute_hour,
            "mean_facility_power_kw": facility_energy / n_hours,
            "peak_facility_power_kw": peak_facility_power,
            "peak_facility_power_hour": peak_facility_hour,
            "daily_peak_valley_facility_power_kw": _mean_daily_peak_valley(hourly_facility),
            "max_hourly_ramp_facility_power_kw": _max_hourly_ramp(hourly_facility),
            "high_carbon_period_facility_energy_share_pct": _safe_scaled_ratio(high_carbon_energy, facility_energy, 100.0),
        }
        record.update(_country_task_type_metrics(loc_rows, n_hours, "training"))
        record.update(_country_task_type_metrics(loc_rows, n_hours, "inference"))
        records.append(record)

    return pd.DataFrame(records).reindex(columns=COUNTRY_SUMMARY_COLUMNS)


def build_utilization_frames(assignment, dc_cfg, n_hours, it_capacity_kw):
    locations = [dc["location"] for dc in dc_cfg]
    capacities = _capacity_matrix(dc_cfg, it_capacity_kw)

    hours = np.repeat(np.arange(int(n_hours), dtype=int), len(locations))
    location_idx = np.tile(np.arange(len(locations), dtype=int), int(n_hours))
    detail = pd.DataFrame(
        {
            "hour": hours,
            "location": np.asarray(locations, dtype=object)[location_idx],
        }
    )

    detail["capacity_it_power_kw"] = np.tile(capacities[:, 0], int(n_hours))
    detail["used_it_power_kw"] = 0.0

    if not assignment.empty:
        rows = assignment.copy()
        rows["hour"] = rows["start_hour"].astype(int)
        rows["used_it_power_kw"] = _it_power_kw(
            rows["cores_req"].astype(float),
            rows["gpu_req"].astype(float),
            rows["mem_req"].astype(float),
        ) * rows["fraction"].astype(float)
        used = rows.groupby(["hour", "location"], as_index=False)["used_it_power_kw"].sum()
        detail = detail.merge(used, on=["hour", "location"], how="left", suffixes=("", "_actual"))
        detail["used_it_power_kw"] = detail["used_it_power_kw_actual"].fillna(0.0)
        detail = detail.drop(columns=["used_it_power_kw_actual"])

    utilization_cols = []
    with np.errstate(divide="ignore", invalid="ignore"):
        detail["utilization_it_power"] = np.where(
            detail["capacity_it_power_kw"].to_numpy(dtype=float) > 0.0,
            detail["used_it_power_kw"].to_numpy(dtype=float) / detail["capacity_it_power_kw"].to_numpy(dtype=float),
            np.nan,
        )
    utilization_cols.append("utilization_it_power")

    summary = {}
    for util_col in utilization_cols:
        values = detail[util_col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            summary[f"{util_col}_max"] = 0.0
            summary[f"{util_col}_p95"] = 0.0
            summary[f"{util_col}_mean"] = 0.0
            summary[f"{util_col}_slots_ge_90pct"] = 0
            summary[f"{util_col}_slots_ge_99pct"] = 0
            continue
        summary[f"{util_col}_max"] = float(np.max(values))
        summary[f"{util_col}_p95"] = float(np.percentile(values, 95))
        summary[f"{util_col}_mean"] = float(np.mean(values))
        summary[f"{util_col}_slots_ge_90pct"] = int(np.sum(values >= 0.90))
        summary[f"{util_col}_slots_ge_99pct"] = int(np.sum(values >= 0.99))

    return detail, summary


def write_utilization_outputs(name, assignment, dc_cfg, n_hours, it_capacity_kw, utilization_summary_path):
    _, summary = build_utilization_frames(
        assignment,
        dc_cfg,
        n_hours,
        it_capacity_kw,
    )
    pd.DataFrame([{"strategy": name, **summary}]).to_csv(
        utilization_summary_path,
        mode="a",
        header=not os.path.exists(utilization_summary_path),
        index=False,
    )


def write_strategy_outputs(name, tasks, assignment, summary, summary_path, task_type_location_path, dc_cfg):
    _with_output_column_labels(
        order_summary_columns(pd.DataFrame([summary])),
        SUMMARY_OUTPUT_COLUMNS,
    ).to_csv(
        summary_path,
        mode="a",
        header=not os.path.exists(summary_path),
        index=False,
    )
    summarize_task_type_by_location(name, tasks, assignment, dc_cfg).to_csv(
        task_type_location_path,
        mode="a",
        header=not os.path.exists(task_type_location_path),
        index=False,
    )
    print(f"Saved strategy '{name}' to summary.")


def write_country_summary_outputs(name, assignment, dc_cfg, n_hours, ci_by_hour_loc, high_carbon_quantile, country_summary_path):
    _with_output_column_labels(
        build_country_summary_frame(
            name,
            assignment,
            dc_cfg,
            n_hours,
            ci_by_hour_loc,
            high_carbon_quantile,
        ),
        COUNTRY_SUMMARY_OUTPUT_COLUMNS,
    ).to_csv(
        country_summary_path,
        mode="a",
        header=not os.path.exists(country_summary_path),
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Self-contained compact hourly Alibaba oracle with carbon, water, transmission, greedy baselines, and MCF.")
    parser.add_argument("--dc-config", default="configs/env/24-datacenters_population_weight_2025.yaml")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--carbon-data-root", default="data/EM-estimate")
    parser.add_argument("--carbon-policy", choices=["CP", "NDC", "NZ"], default="CP")
    parser.add_argument("--water-path", default="data/water_factors_24country.py")
    parser.add_argument("--water-index", type=int, default=None)
    parser.add_argument("--grid-water-policy", choices=["CP", "NDC", "NZ"], default="CP")
    parser.add_argument("--pue-scenario", choices=["Base", "Lift-Off", "High Efficiency", "Headwinds"], default="Base")
    parser.add_argument("--apply-dlc-water-adjustment", action="store_true")
    parser.add_argument("--workload-path", default="data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl")
    parser.add_argument("--task-scale", type=float, default=5.0)
    parser.add_argument("--llm-training-ratio", type=float, default=0.3)
    parser.add_argument("--load-share-mode", choices=["population_weight", "it_ratio_weight"], default="it_ratio_weight")
    parser.add_argument("--cloud-provider", default="azure")
    parser.add_argument("--carbon-weight", type=float, default=1.0)
    parser.add_argument("--water-weight", type=float, default=0.0)
    parser.add_argument("--tx-weight", type=float, default=1.0)
    parser.add_argument("--cpu-kw-per-core", type=float, default=0.006)
    parser.add_argument("--gpu-kw-per-unit", type=float, default=0.25)
    parser.add_argument("--mem-kw-per-gb", type=float, default=0.00007)
    parser.add_argument("--transmission-kwh-per-gb", type=float, default=0.06)
    parser.add_argument("--policy-block-cost", type=float, default=1e9)
    parser.add_argument("--include-transmission-cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-transmission-carbon", action="store_true")
    parser.add_argument("--policy-transmission-matrix", default="data/network_cost/azure_24country_transmission_cost_matrix_2026.csv")
    parser.add_argument("--policy-allowed-mask", default=None)
    parser.add_argument("--objective-normalization", choices=["benchmark", "none"], default="benchmark")
    parser.add_argument("--it-ratio-path", default="data/it_capacity_ratios.py")
    parser.add_argument("--it-capacity-utilization", type=float, default=1.0)
    parser.add_argument("--it-capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--solve", choices=["none", "mcf"], default="none")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--greedy-infeasible-fallback", choices=["none", "constrained"], default="none")
    parser.add_argument("--horizon-hours", type=int, default=8760)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--option-workers", type=int, default=0)
    parser.add_argument("--option-chunk-size", type=int, default=0)
    parser.add_argument("--mcf-backend", choices=["auto", "ortools", "networkx"], default="auto")
    parser.add_argument("--mcf-resource-scale", type=float, default=1_0000_0000.0)
    parser.add_argument("--mcf-cost-scale", type=float, default=1_0000_0000.0)
    parser.add_argument("--mcf-feasibility-tolerance", type=float, default=1e-7)
    parser.add_argument("--write-utilization", action="store_true")
    parser.add_argument("--high-carbon-quantile", type=float, default=0.75)
    parser.add_argument("--transmission-delay-rounding", choices=["none"], default="none")
    parser.add_argument("--output-dir", default="results/global_8760_oracle_it_upper_bound_compact")
    return parser.parse_args()


def main():
    run_start_time = time.perf_counter()
    args = parse_args()
    configure_runtime_parameters(args)

    n_hours = int(args.horizon_hours)
    if n_hours <= 0 or n_hours > 8760:
        raise ValueError("--horizon-hours must be in [1, 8760].")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be a positive integer when provided.")
    if not (0.0 <= float(args.llm_training_ratio) <= 1.0):
        raise ValueError("--llm-training-ratio must be in [0, 1].")
    if min(
        float(args.cpu_kw_per_core),
        float(args.gpu_kw_per_unit),
        float(args.mem_kw_per_gb),
        float(args.transmission_kwh_per_gb),
    ) < 0.0:
        raise ValueError("Energy coefficients must be non-negative.")
    if float(args.policy_block_cost) <= 0.0:
        raise ValueError("--policy-block-cost must be positive.")
    if not (0.0 < float(args.high_carbon_quantile) < 1.0):
        raise ValueError("--high-carbon-quantile must be in (0, 1).")
    if not (0.0 < float(args.it_capacity_utilization) <= 1.0):
        raise ValueError("--it-capacity-utilization must be in (0, 1].")
    if float(args.it_capacity_multiplier) <= 0.0:
        raise ValueError("--it-capacity-multiplier must be positive.")
    if args.water_index is None:
        args.water_index = min(max(int(args.year) - 2025, 0), 5)

    dc_cfg = load_yaml(args.dc_config)["datacenters"]
    locations = [dc["location"] for dc in dc_cfg]
    ci_by_hour_loc = build_carbon_arrays(
        dc_cfg,
        args.year,
        n_hours,
        carbon_data_root=args.carbon_data_root,
        carbon_policy=args.carbon_policy,
    )
    direct_wue_by_hour_loc, grid_water_by_hour_loc, pue_by_hour_loc = build_water_arrays(
        dc_cfg, args.water_path, args.water_index, n_hours,
        apply_dlc_adjustment=args.apply_dlc_water_adjustment,
        pue_scenario=args.pue_scenario,
        grid_water_policy=args.grid_water_policy,
    )

    tasks = load_alibaba_hourly_aggregate_tasks(
        args.workload_path, dc_cfg, args.task_scale, args.year, n_hours,
        float(args.llm_training_ratio), args.load_share_mode, max_tasks=args.max_tasks,
    )
    if tasks.empty:
        raise ValueError("No tasks were loaded.")
    tasks, dropped = filter_tasks_schedulable_within_horizon(tasks, n_hours)
    if tasks.empty:
        raise ValueError("No tasks can be completed within the selected horizon.")
    if dropped:
        print(f"Warning: dropped {dropped} tasks that cannot complete within the selected {n_hours}-hour horizon.", file=sys.stderr)
    tasks = tasks.reset_index(drop=True)
    tasks["task_id"] = np.arange(len(tasks), dtype=int)
    print(f"Prepared hourly aggregate tasks: {len(tasks)} rows in {time.perf_counter() - run_start_time:.1f}s", flush=True)
    if int(args.option_workers) < 0:
        raise ValueError("--option-workers must be non-negative; use 0 for auto.")
    if int(args.option_chunk_size) < 0:
        raise ValueError("--option-chunk-size must be non-negative; use 0 for auto.")
    if float(args.mcf_resource_scale) <= 0.0:
        raise ValueError("--mcf-resource-scale must be positive.")
    if float(args.mcf_cost_scale) <= 0.0:
        raise ValueError("--mcf-cost-scale must be positive.")
    if float(args.mcf_feasibility_tolerance) <= 0.0:
        raise ValueError("--mcf-feasibility-tolerance must be positive.")
    it_capacity_kw, it_ratio_locations, total_it_capacity_kw = build_fixed_it_capacity_kw(
        dc_cfg,
        tasks,
        args.it_ratio_path,
        args.it_capacity_utilization,
        args.it_capacity_multiplier,
    )
    preview = ", ".join(
        f"{dc['location']}={capacity:.3f}"
        for dc, capacity in zip(dc_cfg[:8], it_capacity_kw[:8])
    )
    if len(dc_cfg) > 8:
        preview += ", ..."
    print(
        f"Prepared fixed IT power capacity: total={total_it_capacity_kw:.3f} kW; "
        f"source=alibaba_peak; multiplier={args.it_capacity_multiplier}; {preview}",
        flush=True,
    )

    options = build_options(
        tasks, dc_cfg, ci_by_hour_loc, direct_wue_by_hour_loc, grid_water_by_hour_loc,
        pue_by_hour_loc, args.cloud_provider, args.include_transmission_carbon,
        args.transmission_delay_rounding, n_hours,
        option_workers=args.option_workers,
        option_chunk_size=args.option_chunk_size,
        policy_transmission_matrix_path=args.policy_transmission_matrix,
        policy_allowed_mask_path=args.policy_allowed_mask,
    )
    print(f"Prepared options: {len(options.option_task)} rows in {time.perf_counter() - run_start_time:.1f}s", flush=True)

    options.option_objective = build_objective_values(
        tasks, options, args.carbon_weight, args.water_weight, args.tx_weight,
        args.include_transmission_cost, args.include_transmission_carbon, args.objective_normalization,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"oracle_summary_{timestamp}.csv")
    country_summary_path = os.path.join(args.output_dir, f"oracle_country_summary_{timestamp}.csv")
    task_type_location_path = os.path.join(args.output_dir, f"oracle_task_type_location_ratio_{timestamp}.csv")
    utilization_summary_path = os.path.join(args.output_dir, f"oracle_utilization_summary_{timestamp}.csv")

    summaries = []

    def persist_strategy(name, assignment, start_time=None):
        runtime_seconds = None if start_time is None else time.perf_counter() - float(start_time)
        summary = summarize_assignment(name, assignment, runtime_seconds, n_hours=n_hours)
        write_strategy_outputs(
            name,
            tasks,
            assignment,
            summary,
            summary_path,
            task_type_location_path,
            dc_cfg,
        )
        write_country_summary_outputs(
            name,
            assignment,
            dc_cfg,
            n_hours,
            ci_by_hour_loc,
            args.high_carbon_quantile,
            country_summary_path,
        )
        if args.write_utilization:
            write_utilization_outputs(
                name,
                assignment,
                dc_cfg,
                n_hours,
                it_capacity_kw,
                utilization_summary_path,
            )
        summaries.append(summary)

    if args.run_baselines:
        baseline_specs = build_baseline_specs(
            tasks, options, ci_by_hour_loc, args.carbon_weight, args.water_weight, args.tx_weight,
            args.include_transmission_cost, args.include_transmission_carbon, args.objective_normalization,
        )
        for name, objective_values, method in baseline_specs:
            method_start = time.perf_counter()
            _progress(f"Starting strategy {name} ({method})")
            try:
                if method == "arrival_greedy":
                    solution = _greedy_fractional(
                        tasks, options, dc_cfg, n_hours, objective_values, order_mode="arrival",
                        it_capacity_kw=it_capacity_kw,
                        progress_label=f"{name} greedy",
                    )
                elif method == "constrained_greedy":
                    solution = _greedy_fractional(
                        tasks, options, dc_cfg, n_hours, objective_values, order_mode="constrained",
                        it_capacity_kw=it_capacity_kw,
                        progress_label=f"{name} greedy",
                    )
                elif method == "local_immediate_training_overflow_greedy":
                    solution = solve_local_immediate_training_overflow_greedy(
                        tasks, options, dc_cfg, n_hours,
                        it_capacity_kw=it_capacity_kw,
                        progress_label=f"{name} greedy",
                    )
                elif method == "local_training_time_shift_greedy":
                    solution = solve_local_training_time_shift_greedy(
                        tasks, options, dc_cfg, n_hours, objective_values,
                        it_capacity_kw=it_capacity_kw,
                        progress_label=f"{name} greedy",
                    )
                elif method == "immediate_overflow_greedy":
                    solution = solve_immediate_overflow_greedy(
                        tasks, options, dc_cfg, n_hours, objective_values,
                        it_capacity_kw=it_capacity_kw,
                        progress_label=f"{name} greedy",
                    )
                else:
                    raise ValueError(f"Unknown baseline method: {method}")
                assignment = materialize_assignment(solution, tasks, options, dc_cfg, options.option_objective)
                persist_strategy(name, assignment, method_start)
                _progress(f"Finished strategy {name}", method_start)
            except ValueError as exc:
                if method == "arrival_greedy" and args.greedy_infeasible_fallback == "constrained":
                    fallback_start = time.perf_counter()
                    fallback_name = f"{name}_constrained_fallback"
                    print(
                        f"Warning: {name} is infeasible with arrival order: {exc}. "
                        f"Retrying as {fallback_name}.",
                        file=sys.stderr,
                    )
                    try:
                        solution = _greedy_fractional(
                            tasks, options, dc_cfg, n_hours, objective_values, order_mode="constrained",
                            it_capacity_kw=it_capacity_kw,
                            progress_label=f"{fallback_name} greedy",
                        )
                        assignment = materialize_assignment(solution, tasks, options, dc_cfg, options.option_objective)
                        persist_strategy(fallback_name, assignment, fallback_start)
                        _progress(f"Finished fallback strategy {fallback_name}", fallback_start)
                    except ValueError as fallback_exc:
                        print(
                            f"Warning: skipping {name}; fallback {fallback_name} is also infeasible: {fallback_exc}",
                            file=sys.stderr,
                        )
                        _progress(f"Skipped strategy {name}", method_start)
                else:
                    print(f"Warning: skipping {name} because assignment is infeasible: {exc}", file=sys.stderr)
                    _progress(f"Skipped strategy {name}", method_start)

    if args.solve == "mcf":
        mcf_start = time.perf_counter()
        _progress("Starting single-capacity min-cost flow")
        mcf_solution = solve_single_capacity_min_cost_flow(
            tasks,
            options,
            dc_cfg,
            n_hours,
            objective_values=options.option_objective,
            it_capacity_kw=it_capacity_kw,
            backend=args.mcf_backend,
            resource_scale=args.mcf_resource_scale,
            cost_scale=args.mcf_cost_scale,
            feasibility_tolerance=args.mcf_feasibility_tolerance,
        )
        assignment = materialize_assignment(mcf_solution, tasks, options, dc_cfg)
        persist_strategy("single_capacity_min_cost_flow", assignment, mcf_start)
        _progress("Finished single-capacity min-cost flow", mcf_start)

    if not summaries:
        raise ValueError("No strategies were selected. Use --run-baselines or --solve mcf.")

    summary_df = order_summary_columns(pd.DataFrame(summaries))
    print("Locations:", ", ".join(locations))
    print(f"Hours: {n_hours}; tasks: {len(tasks)}; options: {len(options.option_objective)}")
    print(f"Carbon intensity year: {args.year}")
    print(f"Carbon data root: {args.carbon_data_root}; Carbon policy: {args.carbon_policy}")
    print(f"Water/PUE file: {args.water_path}; Factor index: {args.water_index}; Grid water policy: {args.grid_water_policy}; PUE scenario: {args.pue_scenario}")
    print("Problem granularity: hourly")
    print(f"Task type mode: llm_30_70; LLM training ratio: {args.llm_training_ratio}")
    print("Capacity mode: it_power")
    ratio_preview = ", ".join(
        f"{dc['location']}->{ratio_loc}:{capacity:.3f}kW"
        for dc, ratio_loc, capacity in zip(dc_cfg[:8], it_ratio_locations[:8], it_capacity_kw[:8])
    )
    if len(dc_cfg) > 8:
        ratio_preview += ", ..."
    print(
        f"IT capacity source: alibaba_peak; total: {total_it_capacity_kw:.3f} kW; "
        f"multiplier: {args.it_capacity_multiplier}; fixed per-hour capacities: {ratio_preview}"
    )
    print(f"Objective normalization: {args.objective_normalization}")
    print(f"Transmission cost in objective: {args.include_transmission_cost}")
    print(f"Transmission carbon in objective: {args.include_transmission_carbon}")
    print(f"Transmission delay rounding: {args.transmission_delay_rounding}")
    print(f"High-carbon period threshold: country-year carbon intensity quantile {args.high_carbon_quantile:.2f}")
    print(f"Policy transmission matrix: {args.policy_transmission_matrix or 'none'}")
    print(f"Policy allowed mask: {args.policy_allowed_mask or 'none'}")
    print(f"Greedy infeasible fallback: {args.greedy_infeasible_fallback}")
    print(summary_df.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved country summary: {country_summary_path}")
    print(f"Saved task type/location ratios: {task_type_location_path}")
    if args.write_utilization:
        print(f"Saved utilization summary: {utilization_summary_path}")


if __name__ == "__main__":
    main()
