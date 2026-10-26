"""Etapa 3: exploratory analysis on sample E only (PLANO §7, Etapa 3). Sample M is never read here.

Tables in results/eda/:
- composition.csv: rows per class, category and attack type (the effect of the sampling design);
- feature_summary.csv: per feature, missing and infinite values, range, centre, zeros, distinct values, mode
  share (constant / quasi-constant), skewness (log1p candidates) and IQR outlier share (task 1, 3, 8);
- correlated_pairs.csv: identical columns and pairs above the correlation threshold (task 4);
- structural_checks.json: identities between columns that follow from the window definition;
- number_by_label.csv: `Number` per attack type against the window size of the dataset paper (D12);
- univariate.csv: AUC and KS of each feature alone, benign vs attack, overall and per window group (task 5);
- single_feature_models.csv: MCC of a depth-2 tree on each feature alone, in CV on E (task 7);
- by_type_summary.csv: quartiles of the shortcut candidates per attack type (task 6).
Figures: fig_number_by_type.png, fig_single_feature_mcc.png, fig_distributions.png, fig_spearman.png,
fig_by_type.png. Everything is sorted and seeded, so re-running gives the same files.

Usage: python -m src.data.eda
"""
from itertools import combinations

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.data.labels import BENIGN, window_of
from src.reporting.style import (ATTACK_COLOR, BENIGN_COLOR, DIVERGING, INK_MUTED, SERIES, apply_style, plt,
                                 save)
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

NUMBER, TOT_SUM, AVG, STD, VARIANCE = "Number", "Tot sum", "AVG", "Std", "Variance"
COUNTS = ["ack_count", "syn_count", "fin_count", "rst_count"]
WINDOW = "window"  # packet-window size of each label per the dataset paper (10 or 100)


def finite_values(s: pl.Series) -> np.ndarray:
    return s.drop_nulls().filter(s.drop_nulls().is_finite()).to_numpy()


def as_model_input(s: pl.Series) -> np.ndarray:
    """Nulls and ±inf as NaN (the trees handle NaN); the real treatment is decided in D13."""
    x = s.cast(pl.Float64).to_numpy().astype(float)
    x[~np.isfinite(x)] = np.nan
    return x


def composition(df: pl.DataFrame, levels: list[str]) -> pl.DataFrame:
    parts = [df.group_by(level).agg(pl.len().alias("n_rows"))
             .select(pl.lit(level).alias("level"), pl.col(level).cast(pl.String).alias("value"), "n_rows")
             .sort(["n_rows", "value"], descending=[True, False]) for level in levels]
    return pl.concat(parts).with_columns((pl.col("n_rows") / df.height).round(6).alias("fraction"))


def feature_summary(df: pl.DataFrame, features: list[str], quasi_share: float, skew_threshold: float,
                    categorical: set[str]) -> pl.DataFrame:
    """heavy_tailed: unbounded (max > 1), non-negative, skewed and not categorical; [0, 1] shares are left alone."""
    rows = []
    for c in features:
        x = df[c]
        f = finite_values(x)
        q1, q3 = np.quantile(f, [0.25, 0.75])
        iqr = q3 - q1
        counts = x.value_counts(name="n").sort(["n", c], descending=[True, False], nulls_last=True)
        mode, mode_n = counts.row(0)
        skew = float(stats.skew(f)) if f.std() > 0 else 0.0
        rows.append({
            "feature": c, "n_null": x.null_count(), "n_inf": int(x.drop_nulls().is_infinite().sum()),
            "min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()), "median": float(np.median(f)),
            "std": float(f.std()), "pct_zero": round(100 * float((f == 0).mean()), 4),
            "n_unique": x.drop_nulls().n_unique(), "mode": mode, "mode_share": round(mode_n / df.height, 6),
            "skewness": round(skew, 4),
            "pct_iqr_outliers": round(100 * float(((f < q1 - 1.5 * iqr) | (f > q3 + 1.5 * iqr)).mean()), 4),
            "constant": x.drop_nulls().n_unique() == 1,
            "quasi_constant": mode_n / df.height > quasi_share,
            "heavy_tailed": bool(f.min() >= 0 and f.max() > 1 and skew > skew_threshold and c not in categorical),
        })
    return pl.DataFrame(rows)


def correlated_pairs(df: pl.DataFrame, features: list[str], threshold: float) -> pl.DataFrame:
    finite = df.select([pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in features]).to_pandas()
    pearson, spearman = finite.corr("pearson"), finite.corr("spearman")
    rows = []
    for a, b in combinations(features, 2):
        identical = df.select(pl.col(a).eq_missing(pl.col(b)).all()).item()
        p, s = pearson.loc[a, b], spearman.loc[a, b]
        if identical or abs(p) > threshold or abs(s) > threshold:
            rows.append({"feature_a": a, "feature_b": b, "identical": identical,
                         "pearson": round(float(p), 6), "spearman": round(float(s), 6)})
    schema = {"feature_a": pl.String, "feature_b": pl.String, "identical": pl.Boolean,
              "pearson": pl.Float64, "spearman": pl.Float64}
    return pl.DataFrame(rows, schema=schema), spearman


def structural_checks(df: pl.DataFrame) -> dict:
    rel_diff = lambda a, b: (a - b).abs() / b.abs()  # noqa: E731
    std_null = df[STD].is_null()
    return {
        "counts_never_exceed_Number": {c: df.select((pl.col(c) <= pl.col(NUMBER)).all()).item() for c in COUNTS},
        "count_vs_flag_number_times_Number_max_abs_diff": {
            c: df.select((pl.col(c) - pl.col(c.replace("_count", "_flag_number")) * pl.col(NUMBER)).abs().max()).item()
            for c in COUNTS},
        "ARP_plus_IPv_minus_1_max_abs_diff": df.select((pl.col("ARP") + pl.col("IPv") - 1).abs().max()).item(),
        "Protocol_Type_values": sorted(df["Protocol Type"].unique().to_list()),
        "Tot_sum_vs_AVG_times_Number_max_rel_diff":
            df.select(rel_diff(pl.col(TOT_SUM), pl.col(AVG) * pl.col(NUMBER)).max()).item(),
        "Variance_vs_Std_squared_max_rel_diff":
            df.filter(pl.col(VARIANCE) > 0).select(rel_diff(pl.col(STD) ** 2, pl.col(VARIANCE)).max()).item(),
        "Std_null_iff_Number_is_1": bool((std_null == (df[NUMBER] == 1)).all()),
        "n_Std_null": int(std_null.sum()),
    }


def number_by_label(df: pl.DataFrame, label: str, category: str) -> pl.DataFrame:
    return (df.group_by(label, category, WINDOW)
            .agg(pl.len().alias("n_rows"),
                 (pl.col(NUMBER) == 10).sum().alias("n_number_10"),
                 (pl.col(NUMBER) == 100).sum().alias("n_number_100"),
                 ((pl.col(NUMBER) != 10) & (pl.col(NUMBER) != 100)).sum().alias("n_number_other"),
                 (pl.col(NUMBER) == pl.col(WINDOW)).mean().round(6).alias("share_number_eq_window"))
            .sort([WINDOW, category, label]))


def auc(attack: np.ndarray, benign: np.ndarray) -> float:
    """P(attack value > benign value) + ties/2, i.e. the AUC of the raw feature as a score."""
    u = stats.mannwhitneyu(attack, benign, alternative="two-sided").statistic
    return float(u / (len(attack) * len(benign)))


def univariate(df: pl.DataFrame, features: list[str], target: str) -> pl.DataFrame:
    rows = []
    benign_rows = df.filter(pl.col(target) == 0)
    groups = {"all": df.filter(pl.col(target) == 1),
              "window10": df.filter((pl.col(target) == 1) & (pl.col(WINDOW) == 10)),
              "window100": df.filter((pl.col(target) == 1) & (pl.col(WINDOW) == 100))}
    for c in features:
        b = benign_rows[c].drop_nulls().to_numpy()
        row = {"feature": c}
        for g, attack_rows in groups.items():
            a = attack_rows[c].drop_nulls().to_numpy()
            value = auc(a, b)
            row[f"auc_{g}"] = round(value, 6)
            row[f"separability_{g}"] = round(max(value, 1 - value), 6)
        row["ks_all"] = round(float(stats.ks_2samp(groups["all"][c].drop_nulls().to_numpy(), b).statistic), 6)
        rows.append(row)
    return pl.DataFrame(rows).sort(["separability_all", "feature"], descending=[True, False])


def single_feature_models(df: pl.DataFrame, features: list[str], label: str, target: str, depth: int,
                          n_splits: int, seed: int) -> pl.DataFrame:
    y = df[target].to_numpy()
    window = df[WINDOW].to_numpy()
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, df[label].to_numpy()))
    rows = []
    for c in features:
        x = as_model_input(df[c]).reshape(-1, 1)
        oof = np.empty_like(y)
        mcc = []
        for train, test in folds:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(x[train], y[train])
            oof[test] = tree.predict(x[test])
            mcc.append(matthews_corrcoef(y[test], oof[test]))
        rows.append({"feature": c, "mcc_mean": round(float(np.mean(mcc)), 6), "mcc_sd": round(float(np.std(mcc)), 6),
                     "tpr_window10": round(float(oof[(y == 1) & (window == 10)].mean()), 6),
                     "tpr_window100": round(float(oof[(y == 1) & (window == 100)].mean()), 6),
                     "fpr": round(float(oof[y == 0].mean()), 6)})
    return pl.DataFrame(rows).sort(["mcc_mean", "feature"], descending=[True, False])


def by_type_summary(df: pl.DataFrame, features: list[str], label: str, category: str) -> pl.DataFrame:
    parts = [df.group_by(label, category, WINDOW)
             .agg(pl.len().alias("n_rows"), pl.col(c).quantile(0.25).alias("q25"),
                  pl.col(c).median().alias("median"), pl.col(c).quantile(0.75).alias("q75"))
             .select(pl.lit(c).alias("feature"), pl.all())
             for c in features]
    return pl.concat(parts).sort(["feature", WINDOW, category, label])


def type_order(df: pl.DataFrame, label: str, category: str) -> list[str]:
    """Benign first, then window-10 types, then window-100 types, each by category and name."""
    order = df.select(label, category, WINDOW).unique().sort([WINDOW, category, label])
    labels = order[label].to_list()
    labels.remove(BENIGN)
    return [BENIGN, *labels]


def type_tick(lab: str) -> str:
    return f"{lab} ({window_of(lab)})"


# --- figures --------------------------------------------------------------------------------------------------

def log_scales(summary: pl.DataFrame) -> dict[str, str]:
    """Display scale of heavy-tailed features: log10 if strictly positive, log10(1 + x) if zeros occur."""
    return {r["feature"]: "log10" if r["min"] > 0 else "log10p1"
            for r in summary.filter(pl.col("heavy_tailed")).to_dicts()}


def display(c: str, scales: dict[str, str]):
    if scales.get(c) == "log10":
        return np.log10, "log10(value)"
    if scales.get(c) == "log10p1":
        return (lambda v: np.log10(1 + v)), "log10(1 + value)"
    return (lambda v: v), "value"


def fig_number_by_type(nbl: pl.DataFrame, order: list[str], label: str, target) -> None:
    share = dict(zip(nbl[label], nbl["n_number_100"] / nbl["n_rows"]))
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    ys = np.arange(len(order))[::-1]
    colors = [BENIGN_COLOR if lab == BENIGN else ATTACK_COLOR for lab in order]
    ax.barh(ys, [share[lab] for lab in order], color=colors, height=0.7)
    ax.set_yticks(ys, [type_tick(lab) for lab in order], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of windows with Number = 100")
    ax.set_title("Number by attack type, sample E (window size per the dataset paper)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BENIGN_COLOR), plt.Rectangle((0, 0), 1, 1, color=ATTACK_COLOR)],
              labels=["benign", "attack"], loc="upper right")
    save(fig, target)


def fig_single_feature_mcc(models: pl.DataFrame, candidates: set[str], target) -> None:
    m = models.sort(["mcc_mean", "feature"], descending=[False, True])
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    ys = np.arange(m.height)
    colors = [SERIES[6] if f in candidates else "#a8a7a0" for f in m["feature"]]
    ax.barh(ys, m["mcc_mean"], xerr=m["mcc_sd"], color=colors, height=0.7,
            error_kw={"elinewidth": 0.8, "ecolor": INK_MUTED})
    ax.set_yticks(ys, m["feature"].to_list(), fontsize=7)
    ax.set_xlabel("MCC of a depth-2 tree on one feature (5-fold CV on sample E)")
    ax.set_title("Single-feature models", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[6]), plt.Rectangle((0, 0), 1, 1, color="#a8a7a0")],
              labels=["shortcut candidate (D12)", "other feature"], loc="lower right")
    save(fig, target)


def fig_distributions(df: pl.DataFrame, features: list[str], scales: dict[str, str], target_col: str,
                      target) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2.6 * n_rows))
    for ax, c in zip(axes.flat, features):
        tf, axis_label = display(c, scales)
        values = {}
        for y, name in ((0, "benign"), (1, "attack")):
            values[name] = tf(finite_values(df.filter(pl.col(target_col) == y)[c]))
        lo = min(v.min() for v in values.values())
        hi = max(v.max() for v in values.values())
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 50)
        for name, color in (("benign", BENIGN_COLOR), ("attack", ATTACK_COLOR)):
            ax.hist(values[name], bins=bins, density=True, histtype="step", color=color, linewidth=1.5, label=name)
        ax.set_title(c, loc="left", fontsize=9)
        ax.set_xlabel(axis_label, fontsize=8)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Feature distributions by class, sample E (density; finite values)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, target)


def fig_spearman(spearman, target) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(np.ma.masked_invalid(spearman.to_numpy()), cmap=DIVERGING, vmin=-1, vmax=1)
    ticks = np.arange(len(spearman.columns))
    ax.set_xticks(ticks, spearman.columns, rotation=90, fontsize=6)
    ax.set_yticks(ticks, spearman.columns, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation (grey: constant feature)")
    ax.set_title("Spearman correlation between features, sample E", loc="left")
    save(fig, target)


def fig_by_type(summary: pl.DataFrame, features: list[str], scales: dict[str, str], order: list[str], label: str,
                target) -> None:
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 7.5), sharey=True)
    ys = np.arange(len(order))[::-1]
    for ax, c in zip(np.atleast_1d(axes), features):
        s = {r[label]: r for r in summary.filter(pl.col("feature") == c).to_dicts()}
        tf, axis_label = display(c, scales)
        for y, lab in zip(ys, order):
            color = BENIGN_COLOR if lab == BENIGN else ATTACK_COLOR
            ax.plot([tf(s[lab]["q25"]), tf(s[lab]["q75"])], [y, y], color=color, linewidth=2)
            ax.plot(tf(s[lab]["median"]), y, "o", color=color, markersize=4)
        ax.set_title(c, loc="left", fontsize=9)
        ax.set_xlabel(axis_label, fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [type_tick(lab) for lab in order], fontsize=7)
    fig.suptitle("Median and interquartile range per attack type, sample E (window size in parentheses)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, target)


def main() -> None:
    cfg = load_config()
    cols, e = cfg["columns"], cfg["eda"]
    label, category, target = cols["label"], cols["category"], cols["target"]
    out_dir = path(cfg, "results_dir") / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(path(cfg, "sample_E"))
    df = df.with_columns(pl.col(label).map_elements(window_of, return_dtype=pl.Int64).alias(WINDOW))
    features = [c for c in df.columns if c not in {label, category, target, WINDOW, *cols["provenance"]}]
    apply_style()

    with recorded_run(cfg, out_dir, "eda") as run:
        composition(df, [target, category, label]).write_csv(out_dir / "composition.csv")
        summary = feature_summary(df, features, e["quasi_constant_share"], e["heavy_tail_skew"],
                                  set(e["categorical_features"]))
        summary.write_csv(out_dir / "feature_summary.csv")
        pairs, spearman = correlated_pairs(df, features, e["corr_threshold"])
        pairs.write_csv(out_dir / "correlated_pairs.csv")
        write_json(structural_checks(df), out_dir / "structural_checks.json")
        nbl = number_by_label(df, label, category)
        nbl.write_csv(out_dir / "number_by_label.csv")
        univariate(df, features, target).write_csv(out_dir / "univariate.csv")
        models = single_feature_models(df, features, label, target, e["single_feature_depth"], e["cv_splits"],
                                       cfg["seed"])
        models.write_csv(out_dir / "single_feature_models.csv")
        by_type = by_type_summary(df, e["by_type_features"], label, category)
        by_type.write_csv(out_dir / "by_type_summary.csv")

        scales = log_scales(summary)
        order = type_order(df, label, category)
        candidates = {"Time_To_Live", NUMBER, TOT_SUM, *COUNTS}  # D12: TTL plus the window-scaled features
        fig_number_by_type(nbl, order, label, out_dir / "fig_number_by_type.png")
        fig_single_feature_mcc(models, candidates, out_dir / "fig_single_feature_mcc.png")
        fig_distributions(df, e["distribution_features"], scales, target, out_dir / "fig_distributions.png")
        fig_spearman(spearman, out_dir / "fig_spearman.png")
        fig_by_type(by_type, e["by_type_features"], scales, order, label, out_dir / "fig_by_type.png")

        run["outputs"] = {rel(p): sha256_file(p) for p in sorted(out_dir.iterdir())
                          if p.suffix in {".csv", ".json", ".png"} and not p.name.endswith(".run.json")}
    print(f"wrote {len(run['outputs'])} files to {out_dir}")


if __name__ == "__main__":
    main()
