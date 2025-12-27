#!/usr/bin/env python3
"""
Toy MCMC experiment on the Curie-Weiss model (d=8) for leave-one-out mixing.
Runs a greedy selection (m_keep=7) to choose the coordinate to remove, then
plots TV distance curves for all leave-one-out chains.
"""

import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle


def build_state_space(d):
    states = np.array(list(product([-1, 1], repeat=d)), dtype=np.int8)
    return states


def build_weight_matrix(d):
    idx = np.arange(d)
    diff = np.abs(idx[:, None] - idx[None, :])
    return 2.0 ** (-diff)


def hamiltonian(states, weights, h):
    interaction = np.einsum("ij,ij->i", states @ weights, states)
    field = h * np.sum(states, axis=1)
    return -interaction - field


def stationary_distribution(energies, beta):
    unnorm = np.exp(-beta * energies)
    return unnorm / np.sum(unnorm)


def compute_transition_matrix(states, energies, beta):
    m, d = states.shape
    state_to_index = {tuple(state.tolist()): i for i, state in enumerate(states)}
    P = np.zeros((m, m), dtype=np.float64)
    for i, x in enumerate(states):
        total_rate = 0.0
        for j in range(d):
            y = x.copy()
            y[j] = -y[j]
            k = state_to_index[tuple(y.tolist())]
            delta = energies[k] - energies[i]
            acc = math.exp(-beta * max(delta, 0.0))
            rate = (1.0 / d) * acc
            P[i, k] = rate
            total_rate += rate
        P[i, i] = 1.0 - total_rate
    return P


def aggregate_chain(P, pi, states, keep_indices):
    keep = sorted(keep_indices)
    m = states.shape[0]
    reduced_map = {}
    reduced_states = []
    full_to_reduced = np.empty(m, dtype=np.int32)
    for i, x in enumerate(states):
        key = tuple(x[keep])
        if key not in reduced_map:
            reduced_map[key] = len(reduced_map)
            reduced_states.append(key)
        full_to_reduced[i] = reduced_map[key]
    r = len(reduced_states)
    pi_r = np.zeros(r, dtype=np.float64)
    for i in range(m):
        pi_r[full_to_reduced[i]] += pi[i]
    P_num = np.zeros((r, r), dtype=np.float64)
    for i in range(m):
        ri = full_to_reduced[i]
        for j in range(m):
            rj = full_to_reduced[j]
            P_num[ri, rj] += pi[i] * P[i, j]
    P_r = np.zeros_like(P_num)
    for i in range(r):
        if pi_r[i] > 0:
            P_r[i, :] = P_num[i, :] / pi_r[i]
    return reduced_states, pi_r, P_r


def kl_to_stationary(pi_r, P_r):
    eps = 1e-12
    return np.sum(
        pi_r[:, None]
        * P_r
        * (np.log(P_r + eps) - np.log(pi_r[None, :] + eps))
    )


def tv_distance_curve(P_r, pi_r, n_max):
    r = P_r.shape[0]
    P_pow = np.eye(r, dtype=np.float64)
    pi_row = pi_r[None, :]
    tv = np.zeros(n_max + 1, dtype=np.float64)
    for n in range(n_max + 1):
        tv[n] = 0.5 * np.max(np.sum(np.abs(P_pow - pi_row), axis=1))
        P_pow = P_pow @ P_r
    return tv


def interp_value(tv, x):
    n0 = int(np.floor(x))
    n1 = int(np.ceil(x))
    n0 = max(0, min(n0, len(tv) - 1))
    n1 = max(0, min(n1, len(tv) - 1))
    if n0 == n1:
        return float(tv[n0])
    t = x - n0
    return float((1.0 - t) * tv[n0] + t * tv[n1])


def main():
    d = 8
    beta = 0.1
    h = 1.0
    m_keep = 7
    n_max = 20
    tv_scale = 1000.0
    zoom_half_width = 4
    zoom_pad = 0.004 * tv_scale
    zoom_inner_half_width = 0.02
    zoom_inner_half_width_idx = 1
    zoom_inner_pad = 0.000002 * tv_scale
    output_path = Path(__file__).resolve().parent / "cw_toy_mcmc_tv_zoom.png"

    states = build_state_space(d)
    weights = build_weight_matrix(d)
    energies = hamiltonian(states, weights, h)
    pi = stationary_distribution(energies, beta)
    P = compute_transition_matrix(states, energies, beta)

    if m_keep >= d or m_keep <= 0:
        raise ValueError("m_keep must be between 1 and d-1 for leave-one-out.")
    m_remove = d - m_keep
    if m_remove != 1:
        raise NotImplementedError("This script currently supports leave-one-out only.")

    kl_values = []
    leave_out_data = {}
    tv_curves = {}
    for i in range(d):
        keep = [j for j in range(d) if j != i]
        _, pi_r, P_r = aggregate_chain(P, pi, states, keep)
        kl = kl_to_stationary(pi_r, P_r)
        leave_out_data[i] = (pi_r, P_r)
        kl_values.append(kl)
        print(f"i={i + 1}, D(P^(-i)||Pi^(-i)) = {kl:.6f}")

    i_star = int(np.argmin(kl_values)) + 1
    keep_star = [i for i in range(1, d + 1) if i != i_star]
    print(f"greedy selected remove index = {i_star}")
    print(f"kept coordinates = {keep_star}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for idx in range(1, d + 1):
        pi_r, P_r = leave_out_data[idx - 1]
        tv = tv_distance_curve(P_r, pi_r, n_max) * tv_scale
        tv_curves[idx] = tv
        if idx == i_star:
            ax.plot(range(n_max + 1), tv, label=f"i={idx} (selected)", linewidth=2.4)
        else:
            ax.plot(range(n_max + 1), tv, label=f"i={idx}", linewidth=1.4, alpha=0.9)
    ax.set_xlabel("n (steps)")
    ax.set_ylabel("1000x TV distance")
    ax.set_title("Subset Mixing of Curie-Weiss Model (d=8)", fontsize=12)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    spreads = []
    for n in range(n_max + 1):
        values = [tv_curves[idx][n] for idx in range(1, d + 1)]
        spreads.append(max(values) - min(values))
    n_center = int(np.argmax(spreads))
    zoom_n_min = max(0, n_center - zoom_half_width)
    zoom_n_max = min(n_max, n_center + zoom_half_width)

    zoom_vals = []
    for idx in range(1, d + 1):
        zoom_vals.extend(tv_curves[idx][zoom_n_min : zoom_n_max + 1])
    zoom_vals = np.array(zoom_vals)
    y_min = max(0.0, float(zoom_vals.min()) - zoom_pad)
    y_max = min(tv_scale, float(zoom_vals.max()) + zoom_pad)

    axins = inset_axes(ax, width="42%", height="42%", loc="center right", borderpad=1.2)
    for idx in range(1, d + 1):
        tv = tv_curves[idx]
        if idx == i_star:
            axins.plot(range(n_max + 1), tv, linewidth=2.0)
        else:
            axins.plot(range(n_max + 1), tv, linewidth=1.2, alpha=0.9)
    axins.set_xlim(zoom_n_min, zoom_n_max)
    axins.set_ylim(y_min, y_max)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.12)
    spreads_zoom = spreads[zoom_n_min : zoom_n_max + 1]
    n_center2 = zoom_n_min + int(np.argmax(spreads_zoom))
    zoom2_n_min = max(0, n_center2 - zoom_inner_half_width_idx)
    zoom2_n_max = min(n_max, n_center2 + zoom_inner_half_width_idx)
    zoom2_xlim_min = n_center2 - zoom_inner_half_width
    zoom2_xlim_max = n_center2 + zoom_inner_half_width
    zoom2_vals = []
    for idx in range(1, d + 1):
        tv = tv_curves[idx]
        zoom2_vals.append(interp_value(tv, zoom2_xlim_min))
        zoom2_vals.append(interp_value(tv, zoom2_xlim_max))
        if zoom2_xlim_min <= n_center2 <= zoom2_xlim_max:
            zoom2_vals.append(float(tv[n_center2]))
    zoom2_vals = np.array(zoom2_vals)
    y2_min = max(0.0, float(zoom2_vals.min()) - zoom_inner_pad)
    y2_max = min(tv_scale, float(zoom2_vals.max()) + zoom_inner_pad)

    axins.add_patch(
        Rectangle(
            (zoom2_xlim_min, y2_min),
            zoom2_xlim_max - zoom2_xlim_min,
            y2_max - y2_min,
            fill=False,
            edgecolor="0.5",
            linewidth=0.9,
            zorder=3,
        )
    )

    axins2 = inset_axes(axins, width="35%", height="35%", loc="upper right", borderpad=0.35)
    for idx in range(1, d + 1):
        tv = tv_curves[idx]
        if idx == i_star:
            axins2.plot(range(n_max + 1), tv, linewidth=1.6)
        else:
            axins2.plot(range(n_max + 1), tv, linewidth=1.0, alpha=0.9)
    axins2.set_xlim(zoom2_xlim_min, zoom2_xlim_max)
    axins2.set_ylim(y2_min, y2_max)
    axins2.grid(True, alpha=0.3)
    axins2.tick_params(labelsize=7)
    for spine in axins2.spines.values():
        spine.set_edgecolor("0.3")
        spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=260)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
