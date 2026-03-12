import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


############################################
# token type classifier
############################################

def classify_token(token):
    t = token.strip()

    if t in {"```", "python", "def", "class", "(", ")", "{", "}", ":", "="}:
        return "scaffold"

    if t in {"if", "else", "elif", "for", "while", "return"}:
        return "control"

    if "#" in t or "Example" in t or "usage" in t:
        return "comment"

    if t.replace(".", "").isdigit():
        return "number"

    if t in {"<|endoftext|>", "<|eot_id|>"}:
        return "special"

    if len(t) == 0:
        return "format"

    return "text"


############################################
# load json
############################################

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


############################################
# collect structural token steps
############################################

STRUCT_TYPES = {"scaffold", "control"}

def collect_struct_steps(data):
    steps = []

    for sample in data.values():
        for item in sample:
            tok = item["token"]
            step = item["step"]

            if classify_token(tok) in STRUCT_TYPES:
                steps.append(step)

    return np.array(steps)


############################################
# CDF plot
############################################

def plot_cdf(baseline_steps, forkaware_steps, save_path):

    plt.figure(figsize=(6,5))

    for steps, label in [
        (baseline_steps, "Baseline"),
        (forkaware_steps, "ForkAware")
    ]:

        sorted_steps = np.sort(steps)
        y = np.arange(len(sorted_steps)) / float(len(sorted_steps))

        plt.plot(sorted_steps, y, label=label, linewidth=2)

    plt.xlabel("First-unmask diffusion step")
    plt.ylabel("CDF")
    plt.title("Structural Token Early Commit CDF")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=200)
    plt.show()


############################################
# average step bar plot
############################################

def plot_bar(baseline_steps, forkaware_steps, save_path):

    baseline_mean = baseline_steps.mean()
    forkaware_mean = forkaware_steps.mean()

    plt.figure(figsize=(5,4))

    plt.bar(
        ["Baseline", "ForkAware"],
        [baseline_mean, forkaware_mean],
    )

    plt.ylabel("Average first-unmask step")
    plt.title("Average Structural Commit Step")

    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Baseline mean step:", baseline_mean)
    print("ForkAware mean step:", forkaware_mean)

############################################
# structure-first score
############################################

def compute_structure_first_score(baseline, forkaware):

    leads = []

    for key in baseline.keys():

        b_sample = baseline[key]
        f_sample = forkaware[key]

        # position -> step
        b_map = {x["pos"]: x["step"] for x in b_sample}
        f_map = {x["pos"]: x["step"] for x in f_sample}

        for item in b_sample:

            pos = item["pos"]
            token = item["token"]

            if classify_token(token) not in STRUCT_TYPES:
                continue

            if pos not in f_map:
                continue

            lead = b_map[pos] - f_map[pos]

            leads.append(lead)

    leads = np.array(leads)

    score = np.mean(leads > 0)

    avg_lead = leads.mean()

    print("\nStructure-First Score:", score)
    print("Average lead (steps):", avg_lead)

    return leads

def plot_lead_histogram(leads, save_path):

    plt.figure(figsize=(6,4))

    plt.hist(leads, bins=30)

    plt.axvline(0, linestyle="--")

    plt.xlabel("Baseline step − ForkAware step")
    plt.ylabel("Count")

    plt.title("Structural Token Lead Distribution")

    plt.savefig(save_path, dpi=200)
    plt.show()


############################################
# main
############################################

def main():

    baseline_path = "baseline_first_unmask.json"
    forkaware_path = "forkaware_first_unmask.json"

    out_dir = Path("analysis_figures")
    out_dir.mkdir(exist_ok=True)

    baseline = load_data(baseline_path)
    forkaware = load_data(forkaware_path)

    baseline_struct_steps = collect_struct_steps(baseline)
    forkaware_struct_steps = collect_struct_steps(forkaware)

    plot_cdf(
        baseline_struct_steps,
        forkaware_struct_steps,
        out_dir / "struct_commit_cdf.png",
    )

    plot_bar(
        baseline_struct_steps,
        forkaware_struct_steps,
        out_dir / "struct_commit_bar.png",
    )

    leads = compute_structure_first_score(baseline, forkaware)

    plot_lead_histogram(
        leads,
        out_dir / "structure_lead_hist.png",
    )


if __name__ == "__main__":
    main()