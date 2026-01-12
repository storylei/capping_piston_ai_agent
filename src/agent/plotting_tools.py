"""
Plotting Tools Module (Interpretation-Ready + Robust Time/Group Handling)
Provides visualization capabilities for statistical analysis.

Key design:
- Plot functions MUST return:
  1) 'figure' : matplotlib Figure
  2) 'summary': structured facts for deterministic interpretation
- Agent should NOT ask LLM to infer numbers from a figure.

Key improvements (dataset-agnostic):
1) Robust time axis detection:
   - If a time-like column is numeric (seconds/cycles/sample index), DO NOT parse as datetime.
   - Only parse to datetime for non-numeric columns (strings/datetime).
2) Robust group column resolution:
   - If default group_by doesn't exist, try common label columns (OK_KO_Label, fault, label, class, target, y).
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
    def _resolve_group_column(self, df: pd.DataFrame, group_by: str) -> Optional[str]:
        """Return an existing group/label column name or None (dataset-agnostic)."""
        if group_by in df.columns:
            return group_by
        for cand in ["OK_KO_Label", "fault", "label", "class", "target", "y"]:
            if cand in df.columns:
                return cand
        return None

    def _find_time_axis(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        Try to find a real time axis:
          0) df.attrs['time_column'] if provided
          0.5) common cycle columns: time_cycles/cycle/cycles/time_cycle
          1) DatetimeIndex
          2) datetime dtype column
          3) time-like name hints:
             - if numeric -> treat as numeric axis (DO NOT to_datetime)
             - else -> try pd.to_datetime
        """
        # 0) DataFrame attrs for marked time column
        time_col_attr = getattr(df, "attrs", {}).get("time_column")
        if time_col_attr and time_col_attr in df.columns:
            s = df[time_col_attr]
            # numeric cycles/time
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce")
                if x.notna().sum() > 0:
                    return x, f"Time ({time_col_attr})"
            return s, f"Time ({time_col_attr})"

        # 0.5) common cycle/time columns
        cycle_hints = ["time_cycles", "cycle", "cycles", "time_cycle"]
        for c in df.columns:
            if str(c).lower() in cycle_hints:
                s = df[c]
                x = pd.to_numeric(s, errors="coerce")
                if x.notna().sum() > 0:
                    return x, f"Cycles ({c})"
                return s, f"Cycles ({c})"

        # 1) DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index), "Time (DatetimeIndex)"

        # 2) datetime dtype columns
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if datetime_cols:
            c = datetime_cols[0]
            return df[c], f"Time ({c})"

        # 3) name hints + parseable / numeric-safe
        hints = ["timestamp", "time", "date", "datetime", "elapsed", "duration"]
        hint_cols = [c for c in df.columns if any(h in str(c).lower() for h in hints)]
        for c in hint_cols:
            s = df[c]

            # ✅ numeric time-like axis: keep numeric, DO NOT parse to datetime
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce")
                if x.notna().sum() > 0:
                    return x, f"Time ({c})"

            # ✅ non-numeric: try parse to datetime
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().sum() > 0:
                    return parsed, f"Time ({c})"
            except Exception:
                continue

        return None, None

    def _numeric_stats(self, s: pd.Series) -> Dict[str, Any]:
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
        if s.dtype == "object" or s.dtype.name == "category":
            return True
        if pd.api.types.is_numeric_dtype(s):
            return False
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
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}

            time_values, time_label = self._find_time_axis(df)
            warning = None
            is_true_time_series = True  # Track if this is real time data

            if time_values is None:
                if not allow_sample_index_fallback:
                    return {"success": False, "error": "No real time axis found (DatetimeIndex or time-like column)."}
                time_values = pd.Series(df.index)
                time_label = "Sample Index"
                is_true_time_series = False
                warning = "No real time axis detected; this is an index plot (not a true time series)."

            fig, ax = plt.subplots(figsize=(12, 6))

            summary_group_stats: Dict[str, Any] = {}

            gb = self._resolve_group_column(df, group_by)

            if separate_groups and gb is not None:
                groups = df[gb].dropna().unique()
                for g in groups:
                    mask = df[gb] == g
                    y = pd.to_numeric(df.loc[mask, column], errors="coerce")

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
            # Use appropriate title based on data type
            if is_true_time_series:
                plot_title = title or f"Time Series: {column}"
            else:
                plot_title = title or f"Index Plot: {column}"
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            summary = {
                "plot_type": "time_series" if is_true_time_series else "index_plot",
                "is_true_time_series": is_true_time_series,
                "column": column,
                "x_axis": time_label,
                "has_groups": gb is not None and separate_groups,
                "group_column": gb if (gb is not None and separate_groups) else None,
                "groups": list(summary_group_stats.keys()) if summary_group_stats else [],
                "group_stats": summary_group_stats,
            }
            if warning:
                summary["note"] = warning

            out = {"success": True, "figure": fig, "plot_type": "time_series" if is_true_time_series else "index_plot", "column": column, "summary": summary}
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
        sampling_rate: float = None,  # Changed: None means no real sampling rate provided
        title: str = None,
        top_k_peaks: int = 5,
        is_waveform: bool = None,  # New: explicitly declare if this is true waveform data
    ) -> Dict[str, Any]:
        """
        Plot frequency spectrum (FFT).
        
        IMPORTANT:
        - For true waveform data (sensor readings with known sampling rate), provide sampling_rate in Hz.
        - For feature tables (aggregated/computed features), leave sampling_rate=None.
          Result will be labeled as 'sample-index spectrum' without Hz units.
        """
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}
            
            # Auto-detect if data is waveform or feature table
            if is_waveform is None:
                # Only consider waveform if sampling_rate is explicitly provided AND > 1.0
                # or if df has waveform attribute explicitly set to True
                df_is_waveform = hasattr(df, 'attrs') and df.attrs.get('is_waveform', False)
                has_real_sampling_rate = sampling_rate is not None and sampling_rate > 1.0
                is_waveform = df_is_waveform or has_real_sampling_rate
            
            # Get sampling rate from attrs if not provided
            if sampling_rate is None and hasattr(df, 'attrs') and df.attrs.get('is_waveform'):
                sampling_rate = df.attrs.get('sampling_rate', None)
            
            # If still no sampling rate, default to 1.0 for computation only
            if sampling_rate is None:
                sampling_rate = 1.0

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

                if freqs_pos.size == 0:
                    return freqs_pos, mag_pos, []

                idx = np.argsort(mag_pos)[::-1]
                idx = idx[: min(top_k_peaks, idx.size)]
                peaks = [(float(freqs_pos[i]), float(mag_pos[i])) for i in idx]
                peaks_sorted = sorted(peaks, key=lambda x: x[0])
                return freqs_pos, mag_pos, peaks_sorted

            dominant: Dict[str, Any] = {}

            gb = self._resolve_group_column(df, group_by)

            if gb is not None:
                groups = df[gb].dropna().unique()
                any_plotted = False
                for g in groups:
                    arr = pd.to_numeric(df.loc[df[gb] == g, column], errors="coerce").dropna().values
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

            # Use appropriate labels based on data type
            # STRICT: Only call it waveform if is_waveform=True AND sampling_rate > 1.0
            if is_waveform and sampling_rate > 1.0:
                xlabel = "Frequency (Hz)"
                plot_title = title or f"Frequency Spectrum: {column}"
                plot_type_name = "frequency_spectrum"
                note = f"Real waveform FFT (sampling rate: {sampling_rate} Hz). Dominant frequencies are top-{top_k_peaks} peaks."
                summary_sampling_rate = float(sampling_rate)
            else:
                xlabel = "Sample-Index Frequency"
                plot_title = title or f"Sample-Index Spectrum: {column}"
                plot_type_name = "sample_index_spectrum"
                note = f"⚠️ Feature table spectrum (NOT physical frequency). This shows patterns in sample order, not real Hz. Top-{top_k_peaks} peaks by magnitude."
                summary_sampling_rate = None  # Don't report sampling_rate for feature tables
                is_waveform = False  # Force to False for feature tables
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Magnitude")
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            summary = {
                "plot_type": plot_type_name,
                "is_waveform": is_waveform,
                "column": column,
                "sampling_rate": summary_sampling_rate,  # None for feature tables
                "has_groups": gb is not None,
                "group_column": gb if gb is not None else None,
                "groups": list(dominant.keys()),
                "dominant_peaks": dominant,
                "note": note,
            }

            return {"success": True, "figure": fig, "plot_type": plot_type_name, "column": column, "summary": summary}

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
        try:
            if column not in df.columns:
                return {"success": False, "error": f"Column '{column}' not found in dataset"}

            gb = self._resolve_group_column(df, group_by)
            if gb is None:
                return {"success": False, "error": "No group/label column found for distribution comparison."}
            group_by = gb

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

                    counts = {}
                    for g in df_clean[group_by].unique():
                        sub = df_clean[df_clean[group_by] == g][column]
                        counts[str(g)] = sub.value_counts().to_dict()
                    summary = {
                        "plot_type": "distribution_histogram",
                        "column": column,
                        "is_categorical": True,
                        "has_groups": True,
                        "group_column": group_by,
                        "groups": [str(g) for g in df_clean[group_by].unique()],
                        "category_counts": counts,
                    }

                else:
                    groups = df_clean[group_by].unique()
                    all_vals = pd.to_numeric(df_clean[column], errors="coerce").dropna().values
                    if all_vals.size == 0:
                        return {"success": False, "error": "No valid numeric values to plot."}

                    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

                    for g in groups:
                        vals = pd.to_numeric(df_clean.loc[df_clean[group_by] == g, column], errors="coerce").dropna().values
                        ax.hist(vals, bins=bin_edges, label=str(g), alpha=0.6, edgecolor="black")
                        group_stats[str(g)] = self._numeric_stats(pd.Series(vals))

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
                        "has_groups": True,
                        "group_column": group_by,
                        "groups": [str(g) for g in groups],
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
    # Plot: Feature Comparison
    # -----------------------------
    def plot_feature_comparison(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: str = "OK_KO_Label",
        title: str = None,
        bins: int = 20,
    ) -> Dict[str, Any]:
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
            gb = self._resolve_group_column(df, group_by)

            for i, col in enumerate(valid):
                ax = axes[i]
                col_series = df[col]

                if gb is not None:
                    groups = df[gb].dropna().unique()
                    per_group = {}
                    for g in groups:
                        vals = pd.to_numeric(df.loc[df[gb] == g, col], errors="coerce").dropna()
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
