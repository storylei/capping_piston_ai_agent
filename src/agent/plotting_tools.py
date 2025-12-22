"""
Plotting Tools Module (Interpretation-Ready Version)
Provides visualization capabilities for statistical analysis.

Key design:
- Plot functions MUST return:
  1) 'figure' : matplotlib Figure
  2) 'summary': structured facts for deterministic interpretation
- Agent should NOT ask LLM to infer numbers from a figure.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PlottingTools:
    """Tools for generating various plots + deterministic summaries."""

    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 10

    # -----------------------------
    # Helpers
    # -----------------------------
    def _find_time_axis(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        Try to find a real time axis:
          1) DatetimeIndex
          2) datetime dtype column
          3) column name hints: time/date/timestamp/datetime + parseable
          4) Check for C-MAPSS time_cycles column
        """
        # 0) Check DataFrame attrs for marked time column (C-MAPSS)
        time_col_attr = getattr(df, 'attrs', {}).get('time_column')
        if time_col_attr and time_col_attr in df.columns:
            return df[time_col_attr], f"Time Cycles ({time_col_attr})"
        
        # Also check for common cycle/time columns in industrial data
        cycle_hints = ["time_cycles", "cycle", "cycles", "time_cycle"]
        for c in df.columns:
            if str(c).lower() in cycle_hints:
                return df[c], f"Cycles ({c})"
        
        # 1) DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index), "Time (DatetimeIndex)"

        # 2) datetime dtype columns
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if datetime_cols:
            c = datetime_cols[0]
            return df[c], f"Time ({c})"

        # 3) name hints + parseable
        hints = ["timestamp", "time", "date", "datetime"]
        hint_cols = [c for c in df.columns if any(h in str(c).lower() for h in hints)]
        for c in hint_cols:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    return parsed, f"Time ({c})"
            except Exception:
                continue

        return None, None

    def _group_keys(self, df: pd.DataFrame, group_by: str) -> List[str]:
        if group_by in df.columns:
            return [str(x) for x in df[group_by].dropna().unique().tolist()]
        return []

    def _numeric_stats(self, s: pd.Series) -> Dict[str, Any]:
        """Return deterministic stats for a numeric series."""
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "variance": None,
                "min": None,
                "max": None,
                "q05": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q95": None,
            }
        return {
            "count": int(x.shape[0]),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
            "variance": float(x.var(ddof=1)) if x.shape[0] > 1 else 0.0,
            "min": float(x.min()),
            "max": float(x.max()),
            "q05": float(x.quantile(0.05)),
            "q25": float(x.quantile(0.25)),
            "q50": float(x.quantile(0.50)),
            "q75": float(x.quantile(0.75)),
            "q95": float(x.quantile(0.95)),
        }

    def _is_categorical(self, s: pd.Series) -> bool:
        """Check if a series should be treated as categorical.
        
        For numerical data, we should NOT treat it as categorical just because
        of low unique count - sensor data often has repeated values.
        """
        # Only object/string types are categorical
        if s.dtype == "object" or s.dtype.name == "category":
            return True
        # Numeric types are NEVER categorical (sensors, measurements, etc.)
        if pd.api.types.is_numeric_dtype(s):
            return False
        # Fallback for other types
        return False

    # -----------------------------
    # Plot: Time Series
    # -----------------------------
    def plot_time_series(
        self,
        df: pd.DataFrame,
        column: str,
        group_by: str = "OK_KO_Label",
        title: str = None,
        separate_groups: bool = True,
        allow_sample_index_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Plot time series. Uses real time axis if available, otherwise uses sample index with warning.
        Also returns summary with per-group basic stats (mean/std etc).
        """
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}

            time_values, time_label = self._find_time_axis(df)
            warning = None

            if time_values is None:
                if not allow_sample_index_fallback:
                    return {"success": False, "error": "No real time axis found (DatetimeIndex or datetime column)."}
                time_values = pd.Series(df.index)
                time_label = "Sample Index"
                warning = "No real time axis detected; plotted against sample index (not true time)."

            fig, ax = plt.subplots(figsize=(12, 6))

            # Prepare summary stats
            summary_group_stats: Dict[str, Any] = {}

            if separate_groups and group_by in df.columns:
                groups = df[group_by].dropna().unique()
                for g in groups:
                    mask = df[group_by] == g
                    y = pd.to_numeric(df.loc[mask, column], errors="coerce")
                    # Align x with mask if possible
                    if isinstance(time_values, pd.Series) and len(time_values) == len(df):
                        x = time_values.loc[mask]
                    else:
                        x = df.loc[mask].index

                    valid = y.notna() & pd.Series(x).notna()
                    ax.plot(
                        np.array(pd.Series(x)[valid]),
                        np.array(y[valid]),
                        label=str(g),
                        alpha=0.7,
                        linewidth=1.5,
                    )

                    summary_group_stats[str(g)] = self._numeric_stats(y)

                ax.legend()
            else:
                y = pd.to_numeric(df[column], errors="coerce")
                x = time_values if isinstance(time_values, pd.Series) and len(time_values) == len(df) else pd.Series(df.index)
                valid = y.notna() & pd.Series(x).notna()
                ax.plot(np.array(pd.Series(x)[valid]), np.array(y[valid]), linewidth=1.5)
                summary_group_stats["ALL"] = self._numeric_stats(y)

            ax.set_xlabel(time_label)
            ax.set_ylabel(column)
            ax.set_title(title or f"Time Series Plot: {column}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            summary = {
                "plot_type": "time_series",
                "column": column,
                "x_axis": time_label,
                "group_stats": summary_group_stats,
            }
            if warning:
                summary["note"] = warning

            out = {"success": True, "figure": fig, "plot_type": "time_series", "column": column, "summary": summary}
            if warning:
                out["warning"] = warning
            return out

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # Plot: Frequency Spectrum (FFT)
    # -----------------------------
    def plot_frequency_spectrum(
        self,
        df: pd.DataFrame,
        column: str,
        group_by: str = "OK_KO_Label",
        sampling_rate: float = 1.0,
        title: str = None,
        top_k_peaks: int = 5,
    ) -> Dict[str, Any]:
        """
        Plot frequency spectrum (FFT). Returns dominant frequencies (top peaks) for each group.
        """
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}

            fig, ax = plt.subplots(figsize=(12, 6))

            def compute_fft_peaks(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size < 2:
                    raise ValueError("Not enough valid samples for FFT (need at least 2).")

                n = arr.size
                fft_values = np.fft.fft(arr)
                freqs = np.fft.fftfreq(n, d=1.0 / float(sampling_rate))

                pos = freqs > 0
                freqs_pos = freqs[pos]
                mag_pos = np.abs(fft_values[pos])

                # Peak picking: take top-k magnitudes
                if freqs_pos.size == 0:
                    return freqs_pos, mag_pos, []

                idx = np.argsort(mag_pos)[::-1]
                idx = idx[: min(top_k_peaks, idx.size)]
                peaks = [(float(freqs_pos[i]), float(mag_pos[i])) for i in idx]
                # sort peaks by frequency for readability
                peaks_sorted = sorted(peaks, key=lambda x: x[0])
                return freqs_pos, mag_pos, peaks_sorted

            dominant: Dict[str, Any] = {}

            if group_by in df.columns:
                groups = df[group_by].dropna().unique()
                any_plotted = False
                for g in groups:
                    arr = pd.to_numeric(df.loc[df[group_by] == g, column], errors="coerce").dropna().values
                    if arr.size < 2:
                        continue

                    freqs_pos, mag_pos, peaks = compute_fft_peaks(arr)
                    ax.plot(freqs_pos, mag_pos, label=str(g), alpha=0.7, linewidth=1.5)
                    dominant[str(g)] = peaks
                    any_plotted = True

                if not any_plotted:
                    return {"success": False, "error": "Not enough valid samples for FFT in any group."}

                ax.legend()
            else:
                arr = pd.to_numeric(df[column], errors="coerce").dropna().values
                freqs_pos, mag_pos, peaks = compute_fft_peaks(arr)
                ax.plot(freqs_pos, mag_pos, linewidth=1.5)
                dominant["ALL"] = peaks

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_title(title or f"Frequency Spectrum: {column}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            summary = {
                "plot_type": "frequency_spectrum",
                "column": column,
                "sampling_rate": float(sampling_rate),
                "dominant_frequencies": dominant,
                "note": f"Dominant frequencies are top-{top_k_peaks} peaks by magnitude among positive frequencies.",
            }

            return {"success": True, "figure": fig, "plot_type": "frequency_spectrum", "column": column, "summary": summary}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # Plot: Distribution Comparison
    # -----------------------------
    def plot_distribution_comparison(
        self,
        df: pd.DataFrame,
        column: str,
        group_by: str = "OK_KO_Label",
        plot_type: str = "histogram",
        title: str = None,
        bins: int = 30,
    ) -> Dict[str, Any]:
        """
        Plot distribution comparison between groups.
        Returns summary including per-group stats and (for histogram) bin counts.
        """
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}
            if group_by not in df.columns:
                return {"success": False, "error": f"Group column '{group_by}' not found"}

            plt.close("all")
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            df_clean = df[[column, group_by]].dropna()

            is_cat = self._is_categorical(df_clean[column])

            group_stats: Dict[str, Any] = {}
            histogram_bins: Dict[str, Any] = {}

            if plot_type == "histogram":
                if is_cat:
                    cross_tab = pd.crosstab(df_clean[column], df_clean[group_by])
                    cross_tab.plot(kind="bar", ax=ax, alpha=0.85, edgecolor="black")
                    ax.set_ylabel("Count")
                    ax.set_xlabel(column)
                    ax.legend(title=group_by)
                    plt.xticks(rotation=45)

                    # categorical summary
                    counts = {}
                    for g in df_clean[group_by].unique():
                        sub = df_clean[df_clean[group_by] == g][column]
                        counts[str(g)] = sub.value_counts().to_dict()
                    summary = {
                        "plot_type": "distribution_histogram",
                        "column": column,
                        "is_categorical": True,
                        "category_counts": counts,
                    }

                else:
                    groups = df_clean[group_by].unique()
                    # Use shared bin edges for comparability
                    all_vals = pd.to_numeric(df_clean[column], errors="coerce").dropna().values
                    if all_vals.size == 0:
                        return {"success": False, "error": "No valid numeric values to plot."}

                    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

                    for g in groups:
                        vals = pd.to_numeric(df_clean.loc[df_clean[group_by] == g, column], errors="coerce").dropna().values
                        ax.hist(vals, bins=bin_edges, label=str(g), alpha=0.6, edgecolor="black")
                        group_stats[str(g)] = self._numeric_stats(pd.Series(vals))

                        # Store bin counts
                        counts, edges = np.histogram(vals, bins=bin_edges)
                        histogram_bins[str(g)] = {
                            "bin_edges": [float(x) for x in edges],
                            "bin_counts": [int(x) for x in counts],
                        }

                    ax.legend()
                    ax.set_ylabel("Frequency")
                    ax.set_xlabel(column)

                    summary = {
                        "plot_type": "distribution_histogram",
                        "column": column,
                        "is_categorical": False,
                        "group_stats": group_stats,
                        "histogram_bins": histogram_bins,
                        "note": f"Histogram uses shared bin edges (bins={bins}) for group comparability.",
                    }

            elif plot_type == "kde":
                if is_cat:
                    return {"success": False, "error": "KDE is not suitable for categorical data."}

                groups = df_clean[group_by].unique()
                for g in groups:
                    vals = pd.to_numeric(df_clean.loc[df_clean[group_by] == g, column], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    vals.plot(kind="kde", ax=ax, label=str(g), linewidth=2)
                    group_stats[str(g)] = self._numeric_stats(vals)

                ax.legend()
                ax.set_ylabel("Density")
                ax.set_xlabel(column)

                summary = {
                    "plot_type": "distribution_kde",
                    "column": column,
                    "is_categorical": False,
                    "group_stats": group_stats,
                    "note": "KDE curves are plotted per group; summary includes per-group descriptive stats (not KDE peak).",
                }

            elif plot_type == "boxplot":
                if is_cat:
                    return {"success": False, "error": "Boxplot is not suitable for categorical y-values."}

                df_num = df_clean.copy()
                df_num[column] = pd.to_numeric(df_num[column], errors="coerce")
                df_num = df_num.dropna(subset=[column])

                df_num.boxplot(column=column, by=group_by, ax=ax)
                plt.suptitle("")
                ax.set_xlabel(group_by)
                ax.set_ylabel(column)

                for g in df_num[group_by].unique():
                    vals = df_num.loc[df_num[group_by] == g, column]
                    group_stats[str(g)] = self._numeric_stats(vals)

                summary = {
                    "plot_type": "distribution_boxplot",
                    "column": column,
                    "is_categorical": False,
                    "group_stats": group_stats,
                    "note": "Boxplot visualizes median and IQR; summary provides quantiles per group.",
                }

            elif plot_type == "violin":
                if is_cat:
                    # fallback to bar for categorical
                    cross_tab = pd.crosstab(df_clean[column], df_clean[group_by])
                    cross_tab.plot(kind="bar", ax=ax, alpha=0.85, edgecolor="black")
                    ax.set_ylabel("Count")
                    ax.set_xlabel(column)
                    ax.legend(title=group_by)
                    plt.xticks(rotation=45)

                    counts = {}
                    for g in df_clean[group_by].unique():
                        sub = df_clean[df_clean[group_by] == g][column]
                        counts[str(g)] = sub.value_counts().to_dict()

                    summary = {
                        "plot_type": "distribution_violin",
                        "column": column,
                        "is_categorical": True,
                        "note": "Violin not suitable for categorical; plotted grouped bar counts instead.",
                        "category_counts": counts,
                    }
                else:
                    df_num = df_clean.copy()
                    df_num[column] = pd.to_numeric(df_num[column], errors="coerce")
                    df_num = df_num.dropna(subset=[column])

                    sns.violinplot(data=df_num, x=group_by, y=column, ax=ax)
                    ax.set_xlabel(group_by)
                    ax.set_ylabel(column)

                    for g in df_num[group_by].unique():
                        vals = df_num.loc[df_num[group_by] == g, column]
                        group_stats[str(g)] = self._numeric_stats(vals)

                    summary = {
                        "plot_type": "distribution_violin",
                        "column": column,
                        "is_categorical": False,
                        "group_stats": group_stats,
                        "note": "Violin plot shows distribution shape; summary includes descriptive stats per group.",
                    }
            else:
                return {"success": False, "error": f"Unsupported plot_type '{plot_type}'"}

            ax.set_title(title or f"Distribution Comparison: {column}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            return {
                "success": True,
                "figure": fig,
                "plot_type": f"distribution_{plot_type}",
                "column": column,
                "summary": summary,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # Plot: Feature Comparison (multi-panel)
    # -----------------------------
    def plot_feature_comparison(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: str = "OK_KO_Label",
        title: str = None,
        bins: int = 20,
    ) -> Dict[str, Any]:
        """
        Plot histograms for multiple features. Returns per-feature per-group stats.
        """
        try:
            if not columns:
                return {"success": False, "error": "No columns specified"}

            valid = [c for c in columns if c in df.columns]
            if not valid:
                return {"success": False, "error": "None of the specified columns found in dataset"}

            n = len(valid)
            n_rows = (n + 1) // 2

            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
            if n == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            summary_stats: Dict[str, Any] = {}

            for i, col in enumerate(valid):
                ax = axes[i]
                col_series = df[col]

                if group_by in df.columns:
                    groups = df[group_by].dropna().unique()
                    per_group = {}
                    for g in groups:
                        vals = pd.to_numeric(df.loc[df[group_by] == g, col], errors="coerce").dropna()
                        if not vals.empty:
                            ax.hist(vals, bins=bins, alpha=0.6, label=str(g))
                        per_group[str(g)] = self._numeric_stats(vals)
                    ax.legend()
                    summary_stats[col] = per_group
                else:
                    vals = pd.to_numeric(col_series, errors="coerce").dropna()
                    ax.hist(vals, bins=bins, alpha=0.7)
                    summary_stats[col] = {"ALL": self._numeric_stats(vals)}

                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.set_title(col)
                ax.grid(True, alpha=0.3)

            for j in range(n, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(title or "Feature Comparison", fontsize=14, y=1.00)
            plt.tight_layout()

            summary = {
                "plot_type": "feature_comparison",
                "columns": valid,
                "per_feature_group_stats": summary_stats,
                "note": f"Each subplot is a histogram (bins={bins}). Summary contains descriptive stats.",
            }

            return {"success": True, "figure": fig, "plot_type": "feature_comparison", "columns": valid, "summary": summary}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # Plot: Correlation Heatmap
    # -----------------------------
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        title: str = None,
        annot: bool = True,
    ) -> Dict[str, Any]:
        """
        Plot correlation heatmap with deterministic correlation matrix summary.
        """
        try:
            if columns:
                df_subset = df[columns]
            else:
                df_subset = df.select_dtypes(include=[np.number])

            if df_subset.empty:
                return {"success": False, "error": "No numerical columns found"}

            corr = df_subset.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=annot, fmt=".2f", center=0, ax=ax, square=True)
            ax.set_title(title or "Correlation Heatmap")
            plt.tight_layout()

            summary = {
                "plot_type": "correlation_heatmap",
                "columns": list(corr.columns),
                "correlation_matrix": corr.round(4).to_dict(),
                "note": "Summary includes full correlation matrix (rounded to 4 decimals).",
            }

            return {"success": True, "figure": fig, "plot_type": "correlation_heatmap", "summary": summary}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # -----------------------------
    # Utility: figure to base64
    # -----------------------------
    def fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64
