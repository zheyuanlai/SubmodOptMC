import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

def simulate_path(P, state_vals, steps, initial_state=None):
    d = int(np.log2(P.shape[0]))

    def state_to_index(state):
        return int("".join(map(str, state)), 2)

    def index_to_state(index):
        return list(map(int, f"{index:0{d}b}"))

    if initial_state is None:
        initial_state = np.random.choice(state_vals, size=d)

    current_state = initial_state
    states = [current_state]

    for _ in range(steps):
        current_index = state_to_index(current_state)
        transition_probs = P[current_index, :]
        next_index = np.random.choice(range(len(transition_probs)), p=transition_probs)
        current_state = index_to_state(next_index)
        states.append(current_state)

    return np.array(states)

def plot_sample_paths(original_path, subset_path, subset_indices):
    steps = original_path.shape[0]
    fig, axes = plt.subplots(len(subset_indices), 1, figsize=(10, 4 * len(subset_indices)), sharex=True)

    if len(subset_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(subset_indices):
        axes[i].plot(range(steps), original_path[:, idx], label=f"Original MC (dim {idx})", linestyle="--")
        axes[i].plot(range(steps), subset_path[:, i], label=f"Subset MC (dim {i})", linestyle="-")
        axes[i].set_ylabel(f"State (dim {idx})")
        axes[i].legend()

    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.savefig("sample_paths.png")
    plt.show()