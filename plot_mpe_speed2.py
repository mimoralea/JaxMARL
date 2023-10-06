import matplotlib.pyplot as plt
import jax.numpy as jnp

env_name = {
    "hanabi": "Hanabi",
    "ant_4x2": "MABrax Ant 4x2",
    "overcooked": "Overcooked",
    "2s3z": "SMAX 2s3z"
}

all_results = jnp.load("all_results_2.npy", allow_pickle=True).item()

benchmark_nums = [1, 100, 10000]
for env in ["hanabi", "ant_4x2", "overcooked", "2s3z"]:
    sps = []
    for num_envs in benchmark_nums:
        sps.append(all_results[(env, num_envs)])
    plt.figure(figsize=(5,5))

    plt.plot(benchmark_nums, sps, linestyle='--', marker='o', label="JaxMARL")
    plt.axhline(y=all_results[(f"og_{env}", 1)], color='r', linestyle='--', label="Original")
    plt.legend()

    plt.ylabel("Steps per Second", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Number of parallel environments", fontsize=14)
    plt.title(f"Steps per second for {env_name[env]}", fontsize=14)
    plt.savefig(f"{env}.pdf")