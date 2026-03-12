import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_token(token_str: str) -> str:
    t = token_str.replace("\n", "\\n").strip()

    if t in {"<|endoftext|>", "<|eot_id|>"}:
        return "special"

    if t in {
        "```", "python", "def", "class", "(", ")", "[", "]", "{", "}",
        ":", ",", ".", "=", "==", "!=", "+", "-", "*", "/", "%", "->"
    }:
        return "scaffold"

    if t in {"if", "else", "elif", "for", "while", "return", "try", "except", "with"}:
        return "control"

    if "#" in t or "Example" in t or "usage" in t or "Output" in t:
        return "comment"

    if t == "" or t in {"\\n"}:
        return "format"

    stripped = t.replace(".", "").replace("-", "")
    if stripped.isdigit():
        return "number"

    if any(ch.isalpha() for ch in t):
        return "text"

    return "other"


COLOR_MAP = {
    "scaffold": "#1f77b4",
    "control": "#d62728",
    "comment": "#2ca02c",
    "format": "#9467bd",
    "number": "#ff7f0e",
    "text": "#7f7f7f",
    "special": "#000000",
    "other": "#8c564b",
}


def select_labels(sample_data, max_labels=30):
    priority_order = {
        "control": 0,
        "scaffold": 1,
        "number": 2,
        "comment": 3,
        "special": 4,
        "text": 5,
        "format": 6,
        "other": 7,
    }

    enriched = []
    for item in sample_data:
        cls = classify_token(item["token"])
        enriched.append((priority_order.get(cls, 99), item["step"], item))

    enriched.sort(key=lambda x: (x[0], x[1]))
    chosen = [x[2] for x in enriched[:max_labels]]
    chosen.sort(key=lambda x: (x["step"], x["pos"]))
    return chosen


def plot_sample_compare(
    baseline_data,
    forkaware_data,
    sample_key,
    save_path=None,
    max_labels=30,
    invert_y=True,
):
    baseline = baseline_data[sample_key]
    forkaware = forkaware_data[sample_key]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, data, title in zip(
        axes,
        [baseline, forkaware],
        [f"{sample_key} - Baseline", f"{sample_key} - ForkAware"],
    ):
        steps = [x["step"] for x in data]
        positions = [x["pos"] for x in data]
        tokens = [x["token"] for x in data]
        classes = [classify_token(tok) for tok in tokens]
        colors = [COLOR_MAP[c] for c in classes]

        ax.scatter(steps, positions, s=28, c=colors, alpha=0.85)
        ax.set_xlabel("Diffusion step")
        ax.set_title(title)
        ax.grid(alpha=0.2)

        if invert_y:
            ax.invert_yaxis()

        label_items = select_labels(data, max_labels=max_labels)
        for item in label_items:
            step = item["step"]
            pos = item["pos"]
            token = item["token"].replace("\n", "\\n")
            ax.annotate(
                token,
                (step, pos),
                fontsize=8,
                alpha=0.9,
                xytext=(3, 3),
                textcoords="offset points",
            )

    axes[0].set_ylabel("Token position")

    handles = []
    labels = []
    for cls, color in COLOR_MAP.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=7,
                linestyle="None",
            )
        )
        labels.append(cls)

    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved figure] {save_path}")

    plt.show()


def main():
    baseline_path = "baseline_first_unmask.json"
    forkaware_path = "forkaware_first_unmask.json"
    out_dir = Path("timeline_compare_figs")
    out_dir.mkdir(exist_ok=True)

    baseline_data = load_json(baseline_path)
    forkaware_data = load_json(forkaware_path)

    common_keys = sorted(set(baseline_data.keys()) & set(forkaware_data.keys()))

    for sample_key in common_keys:
        save_path = out_dir / f"{sample_key}_compare.png"
        plot_sample_compare(
            baseline_data=baseline_data,
            forkaware_data=forkaware_data,
            sample_key=sample_key,
            save_path=str(save_path),
            max_labels=30,
        )


if __name__ == "__main__":
    main()