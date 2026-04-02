from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SIRSimulationConfig:
    beta_per_second: float
    gamma_per_second: float
    tau: float
    num_simulations: int
    horizon_seconds: float
    seed: int = 14


def estimate_large_outbreak_probability(
    events: pd.DataFrame,
    config: SIRSimulationConfig,
) -> tuple[float, float, float]:
    _validate_event_frame(events)
    nodes = sorted(set(events["source"]).union(set(events["target"])))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    event_array = _events_to_array(events, node_index)
    rng = np.random.default_rng(config.seed)

    attack_rates: list[float] = []
    peak_prevalence: list[float] = []
    for _ in range(config.num_simulations):
        attack, peak = _run_single_sir(event_array, len(nodes), config, rng)
        attack_rates.append(attack)
        peak_prevalence.append(peak)

    attack_values = np.asarray(attack_rates, dtype=float)
    peak_values = np.asarray(peak_prevalence, dtype=float)
    large_outbreak_probability = float(np.mean(attack_values >= config.tau))
    mean_attack_rate = float(np.mean(attack_values))
    mean_peak_prevalence = float(np.mean(peak_values))
    return large_outbreak_probability, mean_attack_rate, mean_peak_prevalence


def _run_single_sir(
    base_events: np.ndarray,
    num_nodes: int,
    config: SIRSimulationConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    susceptible = np.ones(num_nodes, dtype=bool)
    infectious = np.zeros(num_nodes, dtype=bool)

    patient_zero = int(rng.integers(0, num_nodes))
    susceptible[patient_zero] = False
    infectious[patient_zero] = True
    ever_infected = np.zeros(num_nodes, dtype=bool)
    ever_infected[patient_zero] = True
    infectious_count = 1
    susceptible_count = num_nodes - 1

    if base_events.shape[0] == 0:
        return 1.0 / num_nodes, 1.0 / num_nodes

    event_t = base_events[:, 0]
    event_u = base_events[:, 1].astype(np.int64)
    event_v = base_events[:, 2].astype(np.int64)
    event_duration = base_events[:, 3]
    event_transmission_probability = 1.0 - np.exp(-config.beta_per_second * event_duration)

    base_span = float(event_t.max() - event_t.min())
    if base_span <= 0:
        raise ValueError("event timestamps must span positive time for cycled simulation")

    peak_count = infectious_count
    event_cursor = 0
    cycle = 0
    current_time = 0.0
    while current_time <= config.horizon_seconds and infectious_count > 0:
        raw_t = float(event_t[event_cursor])
        u = int(event_u[event_cursor])
        v = int(event_v[event_cursor])
        transmission_probability = float(event_transmission_probability[event_cursor])
        event_time = raw_t + cycle * base_span
        if event_time > config.horizon_seconds:
            tail_delta = max(0.0, config.horizon_seconds - current_time)
            if tail_delta > 0:
                infectious_count = _apply_recovery(
                    infectious=infectious,
                    gamma_per_second=config.gamma_per_second,
                    delta_t=tail_delta,
                    rng=rng,
                    infectious_count=infectious_count,
                )
            break

        delta_t = max(0.0, event_time - current_time)
        if delta_t > 0:
            infectious_count = _apply_recovery(
                infectious=infectious,
                gamma_per_second=config.gamma_per_second,
                delta_t=delta_t,
                rng=rng,
                infectious_count=infectious_count,
            )
            if infectious_count == 0:
                break
            current_time = event_time

        infectious_count, susceptible_count = _apply_transmission(
            source=u,
            target=v,
            transmission_probability=transmission_probability,
            susceptible=susceptible,
            infectious=infectious,
            ever_infected=ever_infected,
            rng=rng,
            infectious_count=infectious_count,
            susceptible_count=susceptible_count,
        )
        if infectious_count > peak_count:
            peak_count = infectious_count

        event_cursor += 1
        if event_cursor >= base_events.shape[0]:
            event_cursor = 0
            cycle += 1
        if susceptible_count == 0:
            tail_delta = max(0.0, config.horizon_seconds - current_time)
            if tail_delta > 0:
                infectious_count = _apply_recovery(
                    infectious=infectious,
                    gamma_per_second=config.gamma_per_second,
                    delta_t=tail_delta,
                    rng=rng,
                    infectious_count=infectious_count,
                )
            break

    attack_rate = float(np.mean(ever_infected))
    return attack_rate, float(peak_count / num_nodes)


def _apply_transmission(
    source: int,
    target: int,
    transmission_probability: float,
    susceptible: np.ndarray,
    infectious: np.ndarray,
    ever_infected: np.ndarray,
    rng: np.random.Generator,
    infectious_count: int,
    susceptible_count: int,
) -> tuple[int, int]:
    if infectious[source] and susceptible[target]:
        if rng.random() < transmission_probability:
            susceptible[target] = False
            infectious[target] = True
            ever_infected[target] = True
            infectious_count += 1
            susceptible_count -= 1
    if infectious[target] and susceptible[source]:
        if rng.random() < transmission_probability:
            susceptible[source] = False
            infectious[source] = True
            ever_infected[source] = True
            infectious_count += 1
            susceptible_count -= 1
    return infectious_count, susceptible_count


def _apply_recovery(
    infectious: np.ndarray,
    gamma_per_second: float,
    delta_t: float,
    rng: np.random.Generator,
    infectious_count: int,
) -> int:
    if delta_t <= 0:
        return infectious_count
    recovery_probability = 1.0 - math.exp(-gamma_per_second * delta_t)
    if infectious_count <= 0:
        return 0
    infectious_indices = np.flatnonzero(infectious)
    recovered_draw = rng.random(infectious_indices.size) < recovery_probability
    if not np.any(recovered_draw):
        return infectious_count
    recovered_indices = infectious_indices[recovered_draw]
    infectious[recovered_indices] = False
    return int(infectious_count - recovered_indices.size)


def _events_to_array(events: pd.DataFrame, node_index: dict[str, int]) -> np.ndarray:
    sorted_events = events.sort_values("t_start")[["t_start", "source", "target", "duration_seconds"]]
    t0 = float(sorted_events["t_start"].min())
    rows = []
    for row in sorted_events.itertuples(index=False):
        rows.append(
            [
                float(row.t_start) - t0,
                float(node_index[str(row.source)]),
                float(node_index[str(row.target)]),
                float(row.duration_seconds),
            ]
        )
    return np.asarray(rows, dtype=float)


def _validate_event_frame(events: pd.DataFrame) -> None:
    required_columns = {"source", "target", "t_start", "duration_seconds"}
    missing = required_columns.difference(set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("events must be non-empty")
