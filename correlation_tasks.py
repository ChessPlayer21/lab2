"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    matrix = pd.read_csv(data.csv_path, sep="\t")

    male = matrix[matrix["Gender"] == "Male"]
    female = matrix[matrix["Gender"] == "Female"]

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    men_mri_correlation = {
        f: float(male["MRI_Count"].corr(male[f])) for f in features
    }
    women_mri_correlation = {
        f: float(female["MRI_Count"].corr(female[f])) for f in features
    }

    all_pairs = (
        [(f, c) for f, c in men_mri_correlation.items()] +
        [(f, c) for f, c in women_mri_correlation.items()]
    )
    strongest = max(all_pairs, key=lambda x: abs(x[1]))
    strongest_mri_feature = strongest[0]

    return BrainCorrelationSummary(
        men_count=int(len(male)),
        women_count=int(len(female)),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )

def plot_brain_visuals(csv_path: str) -> None:

    df = pd.read_csv(csv_path, sep="\t")

    male = df[df["Gender"] == "Male"]
    female = df[df["Gender"] == "Female"]

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    cols = features + ["MRI_Count"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for ax, f in zip(axes, features):
        ax.scatter(male["MRI_Count"], male[f],
                   alpha=0.6, s=30, label="Мужчины", color="steelblue")
        ax.scatter(female["MRI_Count"], female[f],
                   alpha=0.6, s=30, label="Женщины", color="salmon")
        ax.set_xlabel("MRI_Count")
        ax.set_ylabel(f)
        ax.set_title(f"{f} vs MRI_Count")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("scatter_mri.png", dpi=120, bbox_inches="tight")
    plt.show()

    male_corr = male[cols].corr()
    female_corr = female[cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(male_corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=axes[0],
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Корреляции — Мужчины")

    sns.heatmap(female_corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=axes[1],
                cbar_kws={"shrink": 0.8})
    axes[1].set_title("Корреляции — Женщины")

    plt.tight_layout()
    plt.savefig("heatmap_corr.png", dpi=120, bbox_inches="tight")
    plt.show()

    male_row = male_corr[["MRI_Count"]].drop("MRI_Count").rename(
        columns={"MRI_Count": "Мужчины"})
    female_row = female_corr[["MRI_Count"]].drop("MRI_Count").rename(
        columns={"MRI_Count": "Женщины"})
    combined = pd.concat([male_row, female_row], axis=1)

    plt.figure(figsize=(6, 5))
    sns.heatmap(combined, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    plt.title("Корреляция признаков с MRI_Count")
    plt.tight_layout()
    plt.savefig("heatmap_mri_only.png", dpi=120, bbox_inches="tight")
    plt.show()