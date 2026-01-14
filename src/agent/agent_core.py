"""
Statistical AI Agent Core Module (Stable + Robust Disambiguation Version)
- Deterministic tools produce ALL numbers and plot summaries.
- Agent NEVER asks LLM to compute or invent numeric facts.
- LLM is only used for fallback general chat, not for tool-result generation.

Key improvement:
- Robust column disambiguation for NL-to-plot:
  - When selecting a VALUE/Y column (time series / FFT), avoid time-like columns
    unless user explicitly requests only time-like columns.
"""

import os
import re
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import pandas as pd

from .llm_interface import LLMInterface
from .conversation import ConversationManager
from .plotting_tools import PlottingTools


class StatisticalAgent:
    """
    Stable Statistical AI Agent:
    1) Parse intent (rule-based, deterministic)
    2) Call tools (Python)
    3) Produce explanation ONLY from tool outputs (no hallucination)
    """

    VALUE_COLUMN_KEYWORDS = [
        "signal",
        "sensor",
        "measurement",
        "value",
        "amplitude",
        "vibration",
        "acc",
        "accel",
        "accelerat",
        "gyro",
        "current",
        "voltage",
        "pressure",
        "temp",
        "temperature",
        "speed",
        "vel",
    ]

    TIME_COLUMN_KEYWORDS = [
        "time",
        "timestamp",
        "datetime",
        "date",
        "cycle",
        "period",
        "phase",
        "elapsed",
        "duration",
    ]

    def __init__(
        self,
        llm_backend: str = "ollama",
        llm_model: str = None,
        api_key: str = None,
        enable_llm_fallback_chat: bool = True,
        enable_llm_interpretation: bool = False,
    ):
        self.llm = LLMInterface(backend=llm_backend, model=llm_model, api_key=api_key)
        self.conversation = ConversationManager()
        self.plotter = PlottingTools()

        self.enable_llm_fallback_chat = enable_llm_fallback_chat
        self.enable_llm_interpretation = enable_llm_interpretation

        self.current_data: Optional[pd.DataFrame] = None
        self.data_info: Dict[str, Any] = {}
        self.analysis_results: Dict[str, Any] = {}

        self.tool_functions = self._register_tool_functions()

    # -----------------------------
    # Data context
    # -----------------------------
    def set_data_context(self, df: pd.DataFrame, data_info: Dict = None):
        self.current_data = df
        self.data_info = data_info or {}
        self.conversation.add_context(self._create_data_context_summary())

    def set_analysis_results(self, results: Dict[str, Any]):
        self.analysis_results = results
    
    def _get_global_task_context(self) -> Dict[str, Any]:
        """
        Extract global task context from current data.
        This ensures LLM always knows: task type, label column, groups, etc.
        GENERIC: Works for any binary/multi-class classification dataset.
        """
        context = {
            "has_data": self.current_data is not None,
            "task_type": "unknown",
            "label_column": None,
            "groups": [],
            "sample_counts": {},
            "analysis_goal": None,
        }
        
        if self.current_data is None:
            return context
        
        df = self.current_data
        
        # Auto-detect label column (generic approach)
        label_candidates = ["OK_KO_Label", "label", "target", "class", "y", "fault", "status"]
        label_col = None
        
        for candidate in label_candidates:
            if candidate in df.columns:
                label_col = candidate
                break
        
        if label_col is not None:
            context["task_type"] = "classification"
            context["label_column"] = label_col
            
            # Get unique groups
            groups = df[label_col].dropna().unique().tolist()
            context["groups"] = [str(g) for g in groups]
            
            # Count samples per group
            for g in groups:
                count = int((df[label_col] == g).sum())
                context["sample_counts"][str(g)] = count
            
            # Generate analysis goal description
            if len(groups) == 2:
                context["analysis_goal"] = f"Binary classification: distinguish {groups[0]} from {groups[1]}"
            else:
                context["analysis_goal"] = f"Multi-class classification: {len(groups)} classes"
        
        return context

    def _create_data_context_summary(self) -> str:
        if self.current_data is None:
            return "No data loaded yet."

        df = self.current_data
        parts = [
            f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]
        if "OK_KO_Label" in df.columns:
            ok_count = int((df["OK_KO_Label"] == "OK").sum())
            ko_count = int((df["OK_KO_Label"] == "KO").sum())
            parts.append(f"OK samples: {ok_count}, KO samples: {ko_count}")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            parts.append(f"Numerical columns: {', '.join(num_cols)}")
        return "\n".join(parts)

    # -----------------------------
    # Public API
    # -----------------------------
    def chat(self, user_message: str, stream: bool = False) -> Dict[str, Any]:
        if self.current_data is None:
            return {
                "response": "⚠️ Please load a dataset first.",
                "plots": [],
                "tool_calls": None,
                "tool_results": [],
            }

        self.conversation.add_message("user", user_message)

        intent = self._parse_intent(user_message)

        if intent["type"] == "unknown":
            if self.enable_llm_fallback_chat:
                return self._fallback_chat(user_message)

            example_col = self.current_data.columns[0] if len(self.current_data.columns) > 0 else "column_name"
            return {
                "response": f"⚠️ I couldn't understand the request. Try: 'mean and variance of {example_col}', or 'plot histogram of {example_col}'.",
                "plots": [],
                "tool_calls": None,
                "tool_results": [],
            }

        tool_name = intent["tool"]
        tool_args = intent.get("args", {})
        tool_func = self.tool_functions.get(tool_name)

        if not tool_func:
            return {
                "response": f"⚠️ Tool not found: {tool_name}",
                "plots": [],
                "tool_calls": None,
                "tool_results": [],
            }

        result = tool_func(**tool_args)

        if not result.get("success", False):
            resp = result.get("message") or f"❌ Tool error: {result.get('error', 'Unknown error')}"
            self.conversation.add_message("assistant", resp, {"tool_results": [result]})
            return {
                "response": resp,
                "plots": [],
                "tool_calls": None,
                "tool_results": [result],
            }

        response = self._render_response_from_tool(result, intent)

        plots = []
        if result.get("plot") is not None:
            plots.append(result["plot"])

        self.conversation.add_message("assistant", response, {"tool_results": [result]})
        return {
            "response": response,
            "plots": plots,
            "tool_calls": None,
            "tool_results": [result],
        }

    # -----------------------------
    # Intent parsing (deterministic)
    # -----------------------------
    def _match_columns(self, message: str) -> List[str]:
        """Word-boundary match to avoid substring false positives."""
        assert self.current_data is not None
        text = message.lower()
        matched = []
        for col in self.current_data.columns:
            col_l = str(col).lower()
            if re.search(r"\b" + re.escape(col_l) + r"\b", text):
                matched.append(col)
        return matched

    def _identify_time_like_columns(self) -> List[str]:
        if self.current_data is None:
            return []

        df = self.current_data
        time_like: List[str] = []

        time_attr = getattr(df, "attrs", {}).get("time_column")
        if time_attr and time_attr in df.columns:
            time_like.append(time_attr)

        for col in df.columns:
            col_lower = str(col).lower()
            if any(key in col_lower for key in self.TIME_COLUMN_KEYWORDS):
                if col not in time_like:
                    time_like.append(col)
                continue

            series = df[col]
            if np.issubdtype(series.dtype, np.datetime64):
                if col not in time_like:
                    time_like.append(col)

        return time_like

    def _prioritize_columns(self, columns: List[str], prefer_keywords: Optional[List[str]] = None) -> List[str]:
        if not prefer_keywords:
            return columns

        keywords = [k.lower() for k in prefer_keywords]
        preferred: List[str] = []
        others: List[str] = []
        for col in columns:
            name = str(col).lower()
            if any(key in name for key in keywords):
                preferred.append(col)
            else:
                others.append(col)
        return preferred + others

    def _collect_feature_names(self, source: Any) -> List[str]:
        names: List[str] = []
        if not source:
            return names

        if isinstance(source, list):
            for item in source:
                if isinstance(item, dict) and "feature" in item:
                    names.append(item["feature"])
                elif isinstance(item, str):
                    names.append(item)
        elif isinstance(source, dict):
            for value in source.values():
                if isinstance(value, (str, list, dict)):
                    names.extend(self._collect_feature_names(value))
        return names

    def _get_analysis_based_columns(
        self,
        limit: int = 3,
        exclude_time_like: bool = True,
        prefer_keywords: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.analysis_results or self.current_data is None:
            return []

        candidates: List[str] = []

        def add_candidates(items: Any):
            for name in self._collect_feature_names(items):
                if name in self.current_data.columns and name not in candidates:
                    candidates.append(name)

        # Try multiple paths for feature ranking
        # Path 1: Top-level feature_ranking
        add_candidates(self.analysis_results.get("feature_ranking"))
        
        # Path 2: ML feature importance
        ml_results = self.analysis_results.get("ml_feature_importance")
        if isinstance(ml_results, dict):
            feature_importance = ml_results.get("feature_importance") or {}
            add_candidates(feature_importance.get("feature_ranking"))
        
        # Path 3: Legacy paths
        stat_analysis = self.analysis_results.get("statistical_analysis")
        if isinstance(stat_analysis, dict):
            add_candidates(stat_analysis.get("feature_ranking"))

        summary = self.analysis_results.get("summary")
        if isinstance(summary, dict):
            add_candidates(summary.get("top_statistical_features"))
            add_candidates(summary.get("top_ml_features"))
            add_candidates(summary.get("consensus_features"))
        
        # Path 4: Direct feature_importance key (backward compatibility)
        legacy_fi = self.analysis_results.get("feature_importance")
        if isinstance(legacy_fi, dict):
            nested_fi = legacy_fi.get("feature_importance") or {}
            add_candidates(nested_fi.get("feature_ranking"))

        if exclude_time_like:
            time_like = set(self._identify_time_like_columns())
            candidates = [c for c in candidates if c not in time_like]

        candidates = self._prioritize_columns(candidates, prefer_keywords)
        return candidates[:limit]

    def _get_default_columns(
        self,
        limit: int = 1,
        exclude_time_like: bool = True,
        prefer_keywords: Optional[List[str]] = None,
    ) -> List[str]:
        if self.current_data is None:
            return []

        defaults: List[str] = []

        defaults.extend(
            self._get_analysis_based_columns(
                limit=limit * 2,
                exclude_time_like=exclude_time_like,
                prefer_keywords=prefer_keywords,
            )
        )

        numeric_cols = [
            col
            for col in self.current_data.select_dtypes(include=[np.number]).columns
            if col != "OK_KO_Label"
        ]

        if exclude_time_like:
            time_like = set(self._identify_time_like_columns())
            numeric_cols = [col for col in numeric_cols if col not in time_like]

        for col in numeric_cols:
            if col not in defaults:
                defaults.append(col)

        defaults = self._prioritize_columns(defaults, prefer_keywords)

        if not defaults and exclude_time_like:
            return self._get_default_columns(limit=limit, exclude_time_like=False, prefer_keywords=prefer_keywords)

        return defaults[:limit]

    def _resolve_columns(
        self,
        matched: List[str],
        limit: int = 1,
        exclude_time_like: bool = True,
        prefer_keywords: Optional[List[str]] = None,
        purpose: str = "y",  # "y" | "x" | "any"
    ) -> List[str]:
        """
        Robust disambiguation:
        - If user message matched some columns, treat them as candidates.
        - For value/y selection (time series / FFT), avoid time-like columns unless ONLY time-like columns were matched.
        This is dataset-agnostic.
        """
        if self.current_data is None:
            return []

        if matched:
            if purpose == "y":
                time_like = set(self._identify_time_like_columns())
                non_time = [c for c in matched if c not in time_like]
                if non_time:
                    matched = non_time  # prefer non-time if available

            matched = self._prioritize_columns(matched, prefer_keywords)
            return matched[:limit]

        return self._get_default_columns(
            limit=limit,
            exclude_time_like=exclude_time_like,
            prefer_keywords=prefer_keywords,
        )

    def _extract_metrics(self, text: str) -> Optional[List[str]]:
        metric_map = {
            "mean": ["mean", "average", "avg"],
            "median": ["median"],
            "mode": ["mode"],
            "std": ["std", "standard deviation", "deviation"],
            "variance": ["variance", "var"],
            "min": ["min", "minimum"],
            "max": ["max", "maximum"],
            "count": ["count"],
        }
        requested = []
        for m, keys in metric_map.items():
            if any(k in text for k in keys):
                requested.append(m)

        if any(k in text for k in ["summary", "statistics", "statistical summary"]) and not requested:
            return None

        return sorted(set(requested)) if requested else None

    def _parse_intent(self, user_message: str) -> Dict[str, Any]:
        text = user_message.lower()
        cols = self._match_columns(user_message)

        if any(k in text for k in ["feature importance", "importance ranking", "rank features", "most important features"]):
            return {"type": "tool", "tool": "get_feature_importance", "args": {"top_n": 10}}

        fft_keywords = ["fft", "frequency spectrum", "fourier", "spectrum"]
        if any(k in text for k in fft_keywords):
            resolved_cols = self._resolve_columns(
                cols,
                limit=1,
                exclude_time_like=True,
                prefer_keywords=self.VALUE_COLUMN_KEYWORDS,
                purpose="y",
            )
            if not resolved_cols:
                return {"type": "unknown"}
            return {"type": "tool", "tool": "plot_frequency_spectrum", "args": {"column": resolved_cols[0]}}

        filter_group = None
        if " ok " in f" {text} " or "ok samples" in text or "ok group" in text or "for ok" in text:
            filter_group = "OK"
        elif " ko " in f" {text} " or "ko samples" in text or "ko group" in text or "for ko" in text:
            filter_group = "KO"

        if "time series" in text or "timeseries" in text:
            resolved_cols = self._resolve_columns(
                cols,
                limit=1,
                exclude_time_like=True,
                prefer_keywords=self.VALUE_COLUMN_KEYWORDS,
                purpose="y",
            )
            if not resolved_cols:
                return {"type": "unknown"}
            return {
                "type": "tool",
                "tool": "plot_time_series",
                "args": {"column": resolved_cols[0], "separate_groups": True, "filter_group": filter_group},
            }

        if any(k in text for k in ["histogram", "distribution", "boxplot", "box plot", "violin", "kde", "density"]):
            resolved_cols = self._resolve_columns(cols, limit=1, purpose="any")
            if not resolved_cols:
                return {"type": "unknown"}
            plot_type = "histogram"
            if "boxplot" in text or "box plot" in text:
                plot_type = "boxplot"
            elif "violin" in text:
                plot_type = "violin"
            elif "kde" in text or "density" in text:
                plot_type = "kde"
            return {
                "type": "tool",
                "tool": "plot_distribution",
                "args": {"column": resolved_cols[0], "plot_type": plot_type, "filter_group": filter_group},
            }

        stat_words = ["mean", "median", "mode", "variance", "std", "standard deviation", "summary", "statistics", "min", "max", "count", "average"]
        if any(w in text for w in stat_words):
            metrics = self._extract_metrics(text)
            return {
                "type": "tool",
                "tool": "get_statistical_summary",
                "args": {"columns": cols if cols else None, "group_by_ok_ko": True, "metrics": metrics},
            }

        if "compare" in text:
            resolved_cols = self._resolve_columns(cols, limit=6, purpose="any")
            if len(resolved_cols) >= 2:
                return {"type": "tool", "tool": "compare_features", "args": {"columns": resolved_cols[:6]}}

        return {"type": "unknown"}

    # -----------------------------
    # Rendering (NO LLM NUMBERS)
    # -----------------------------
    def _render_response_from_tool(self, tool_result: Dict[str, Any], intent: Dict[str, Any]) -> str:
        parts = []

        if tool_result.get("message"):
            parts.append(tool_result["message"].strip())

        summary = tool_result.get("summary")
        if isinstance(summary, dict) and summary:
            parts.append("\n---\n")
            parts.append(self._explain_plot_summary(summary))

        if tool_result.get("warning"):
            parts.append(f"\n⚠️ {tool_result['warning']}")

        if self.enable_llm_interpretation:
            base_response = "\n".join([p for p in parts if p]).strip()
            llm_interpretation = self._llm_interpret_result(tool_result, intent, base_response)
            if llm_interpretation:
                parts.append("\n---\n")
                parts.append("🤖 **AI Analysis:**")
                parts.append(llm_interpretation)

        return "\n".join([p for p in parts if p]).strip()

    def _explain_plot_summary(self, summary: Dict[str, Any]) -> str:
        plot_type = summary.get("plot_type", "plot")
        column = summary.get("column", "feature")

        lines = [
            "🧾 **Plot Interpretation (from tool summary)**",
            f"- Plot type: **{plot_type}**",
            f"- Column: **{column}**",
        ]

        # X axis info (for time series / index plots)
        if "x_axis" in summary:
            lines.append(f"- X axis: {summary['x_axis']}")
        
        # Check if it's a true time series or index plot
        if "is_true_time_series" in summary:
            if summary["is_true_time_series"]:
                lines.append("- ✅ True time series (real time axis detected)")
            else:
                lines.append("- ⚠️ Index plot (no real time axis, using sample index)")
        
        # Waveform vs feature table for FFT
        if "is_waveform" in summary:
            if summary["is_waveform"] and summary.get("sampling_rate"):
                lines.append(f"- ✅ Real waveform data (sampling rate: {summary['sampling_rate']} Hz)")
            else:
                lines.append("- ⚠️ Feature table data (NOT physical waveform)")
        
        # Group information
        if summary.get("has_groups"):
            group_col = summary.get("group_column", "group")
            groups = summary.get("groups", [])
            lines.append(f"- Groups: {', '.join(groups)} (by {group_col})")

        # Group statistics
        if "group_stats" in summary:
            gs = summary["group_stats"]
            lines.append("\n**Statistics by group:**")
            for g, st in gs.items():
                if isinstance(st, dict):
                    stats_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                          for k, v in st.items() if k in ['count', 'mean', 'std', 'min', 'max']])
                    lines.append(f"- {g}: {stats_str}")

        # FFT peaks (renamed from dominant_frequencies)
        if "dominant_peaks" in summary:
            dpeaks = summary["dominant_peaks"]
            is_waveform = summary.get("is_waveform", False)
            unit = "Hz" if is_waveform else "sample-index"
            lines.append(f"\n**Dominant peaks ({unit}):**")
            for g, peaks in dpeaks.items():
                if peaks:
                    peak_str = ", ".join([f"{freq:.2f}" for freq, mag in peaks[:3]])
                    lines.append(f"- {g}: {peak_str}")

        # Note/warning
        if summary.get("note"):
            lines.append(f"\n**Note:** {summary['note']}")

        return "\n".join(lines)

    def _llm_interpret_result(self, tool_result: Dict[str, Any], intent: Dict[str, Any], base_response: str) -> str:
        try:
            tool_name = intent.get("tool", "unknown")
            summary = tool_result.get("summary", {})
            
            # Get global task context (ALWAYS include this)
            task_context = self._get_global_task_context()

            # Build structured context from summary
            context_parts = [f"Tool used: {tool_name}"]
            
            # ALWAYS add global task context first
            if task_context["has_data"]:
                context_parts.append("\n**Dataset Context (CRITICAL - READ THIS FIRST):**")
                context_parts.append(f"- Task type: {task_context['task_type']}")
                if task_context["label_column"]:
                    context_parts.append(f"- Label column: {task_context['label_column']}")
                    context_parts.append(f"- Groups: {', '.join(task_context['groups'])}")
                    context_parts.append(f"- Sample counts: {task_context['sample_counts']}")
                    context_parts.append(f"- Analysis goal: {task_context['analysis_goal']}")
            
            # Add summary fields
            if summary:
                context_parts.append("\n**Analysis Results:**")
                context_parts.append(f"- Plot type: {summary.get('plot_type', 'unknown')}")
                context_parts.append(f"- Column: {summary.get('column', 'unknown')}")
                
                # Time series specifics
                if "is_true_time_series" in summary:
                    context_parts.append(f"- Is true time series: {summary['is_true_time_series']}")
                    if not summary['is_true_time_series']:
                        context_parts.append("  ⚠️ This is an INDEX PLOT, NOT a time series!")
                
                # FFT specifics
                if "is_waveform" in summary:
                    context_parts.append(f"- Is real waveform: {summary['is_waveform']}")
                    if summary.get('sampling_rate'):
                        context_parts.append(f"- Sampling rate: {summary['sampling_rate']} Hz")
                    if not summary['is_waveform']:
                        context_parts.append("  ⚠️ This is a FEATURE TABLE spectrum, NOT physical frequency!")
                
                # Group information
                if summary.get("has_groups"):
                    context_parts.append(f"- Result has groups: Yes ({summary.get('group_column', 'unknown')})")
                    context_parts.append(f"- Groups in result: {', '.join(summary.get('groups', []))}")
                    
                    # Group statistics
                    if "group_stats" in summary:
                        context_parts.append("\n**Group Statistics (USE THESE FOR COMPARISON):**")
                        for group, stats in summary['group_stats'].items():
                            if isinstance(stats, dict):
                                stats_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                                      for k, v in stats.items()])
                                context_parts.append(f"- {group}: {stats_str}")
                
                # Note/warning
                if summary.get("note"):
                    context_parts.append(f"\n**Note:** {summary['note']}")
            
            context = "\n".join(context_parts)

            interpretation_prompt = f"""You are an expert data analyst interpreting results.

DATASET INFORMATION IS PROVIDED ABOVE - READ IT CAREFULLY!

CRITICAL INSTRUCTIONS:
1. Base your interpretation ONLY on the structured data above.
2. DO NOT say "no group information provided" - the groups are ALWAYS listed above.
3. DO NOT invent or calculate any numbers - use only what's given.
4. If "is_true_time_series: False", call it "index plot", NOT "time series".
5. If "is_waveform: False", DO NOT interpret as physical Hz.
6. ALWAYS compare groups when group statistics are provided.

Structured Data:

{context}

Provide a brief (2-3 sentences) expert interpretation:
1. What do these numbers mean in the context of the data?
2. If groups exist, is there a meaningful difference between them?
3. What actionable insight can be drawn?

IMPORTANT: Only explain the numbers shown above. Do NOT invent new statistics."""
            messages = [
                {"role": "system", "content": "You are a data analysis expert. Interpret analysis results concisely."},
                {"role": "user", "content": interpretation_prompt},
            ]

            llm_response = self.llm.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                tools=None,
            )

            interpretation = (llm_response.get("content") or "").strip()
            return interpretation if interpretation else None

        except Exception:
            return None

    # -----------------------------
    # LLM fallback chat (no numbers)
    # -----------------------------
    def _fallback_chat(self, user_message: str) -> Dict[str, Any]:
        messages = self.conversation.get_messages_for_llm()

        example_col = "column_name"
        if self.current_data is not None and len(self.current_data.columns) > 0:
            cols = [c for c in self.current_data.columns if c != "OK_KO_Label"]
            example_col = cols[0] if cols else self.current_data.columns[0]

        system_prompt = (
            f"You are a helpful assistant for analyzing datasets with labels. "
            "Do NOT compute or invent any numeric results. "
            f"If user asks for statistics or plots, instruct them how to ask using tool keywords "
            f"like: 'mean variance {example_col}', 'histogram {example_col}', 'fft {example_col}', 'time series for KO samples'."
        )
        messages.append({"role": "system", "content": system_prompt})

        llm_response = self.llm.generate(messages=messages, temperature=0.6, max_tokens=500, tools=None)
        content = (llm_response.get("content") or "").strip()

        if not content:
            content = f"I couldn't parse that. Try: 'mean and variance of {example_col}', or 'plot histogram of {example_col}'."

        self.conversation.add_message("assistant", content)
        return {"response": content, "plots": [], "tool_calls": None, "tool_results": []}

    # -----------------------------
    # Tool registration
    # -----------------------------
    def _register_tool_functions(self) -> Dict[str, Callable]:
        return {
            "get_statistical_summary": self._tool_get_statistical_summary,
            "plot_time_series": self._tool_plot_time_series,
            "plot_frequency_spectrum": self._tool_plot_frequency_spectrum,
            "plot_distribution": self._tool_plot_distribution,
            "get_feature_importance": self._tool_get_feature_importance,
            "compare_features": self._tool_compare_features,
        }

    # -----------------------------
    # Tools (deterministic)
    # -----------------------------
    def _tool_get_statistical_summary(
        self,
        columns: List[str] = None,
        group_by_ok_ko: bool = True,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None

        if columns is None or len(columns) == 0:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        columns = [c for c in columns if c in df.columns and c != "OK_KO_Label"]
        if not columns:
            return {"success": False, "message": "⚠️ No valid numerical columns found.", "data": {}}

        allowed = {"count", "mean", "median", "mode", "std", "variance", "min", "max"}
        if metrics is not None:
            metrics = [m for m in metrics if m in allowed]
            if not metrics:
                metrics = None

        def calc(series: pd.Series) -> Dict[str, Any]:
            s = series.dropna()
            out: Dict[str, Any] = {}
            if metrics is None or "count" in metrics:
                out["count"] = int(s.shape[0])
            if metrics is None or "mean" in metrics:
                out["mean"] = float(s.mean()) if len(s) else None
            if metrics is None or "median" in metrics:
                out["median"] = float(s.median()) if len(s) else None
            if metrics is None or "mode" in metrics:
                md = s.mode()
                out["mode"] = float(md.iloc[0]) if (len(md) and pd.notna(md.iloc[0])) else None
            if metrics is None or "std" in metrics:
                out["std"] = float(s.std()) if len(s) else None
            if metrics is None or "variance" in metrics:
                out["variance"] = float(s.var()) if len(s) else None
            if metrics is None or "min" in metrics:
                out["min"] = float(s.min()) if len(s) else None
            if metrics is None or "max" in metrics:
                out["max"] = float(s.max()) if len(s) else None
            return out

        summary: Dict[str, Any] = {}
        if group_by_ok_ko and "OK_KO_Label" in df.columns:
            for col in columns:
                summary[col] = {
                    "OK": calc(df.loc[df["OK_KO_Label"] == "OK", col]),
                    "KO": calc(df.loc[df["OK_KO_Label"] == "KO", col]),
                }
        else:
            for col in columns:
                summary[col] = calc(df[col])

        metric_line = ", ".join(metrics) if metrics else "count, mean, median, mode, std, variance, min, max"
        msg = [f"📊 **Statistical Summary** (metrics: {metric_line})\n"]
        for col, st in summary.items():
            msg.append(f"### {col}")
            if isinstance(st, dict) and "OK" in st and "KO" in st:
                msg.append(f"- OK: {st['OK']}")
                msg.append(f"- KO: {st['KO']}")
            else:
                msg.append(f"- {st}")
            msg.append("")

        # Get task context for LLM interpretation
        task_context = self._get_global_task_context()
        
        return {
            "success": True, 
            "message": "\n".join(msg).strip(), 
            "data": summary,
            "summary": {
                "plot_type": "statistical_summary",
                "columns": columns,
                "has_groups": group_by_ok_ko and task_context.get("label_column") is not None,
                "group_column": task_context.get("label_column"),
                "groups": task_context.get("groups", []),
                "metrics": metrics if metrics else ["count", "mean", "median", "mode", "std", "variance", "min", "max"],
                "group_stats": summary,
                "note": f"Statistical comparison across {len(columns)} features."
            }
        }

    def _tool_plot_time_series(self, column: str, separate_groups: bool = True, filter_group: str = None) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None

        plot_df = df
        group_label = ""
        if filter_group and "OK_KO_Label" in df.columns:
            plot_df = df[df["OK_KO_Label"] == filter_group].copy()
            group_label = f" ({filter_group} samples only)"
            separate_groups = False

        res = self.plotter.plot_time_series(plot_df, column=column, separate_groups=separate_groups)

        if not res.get("success", False):
            return {"success": False, "message": f"❌ {res.get('error', 'Time series failed')}"}

        return {
            "success": True,
            "message": f"✅ Generated time series plot for **{column}**{group_label}",
            "plot": res["figure"],
            "warning": res.get("warning"),
            "summary": res.get("summary", {"plot_type": "time_series", "column": column, "note": res.get("warning")}),
        }

    def _tool_plot_frequency_spectrum(self, column: str, sampling_rate: float = 1.0) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None

        res = self.plotter.plot_frequency_spectrum(df, column=column, sampling_rate=sampling_rate)

        if not res.get("success", False):
            return {"success": False, "message": f"❌ {res.get('error', 'FFT failed')}"}

        return {
            "success": True,
            "message": f"✅ Generated frequency spectrum (FFT) plot for **{column}**",
            "plot": res["figure"],
            "summary": res.get("summary", {"plot_type": "frequency_spectrum", "column": column}),
        }

    def _tool_plot_distribution(self, column: str, plot_type: str = "histogram", filter_group: str = None) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None

        plot_df = df
        group_label = ""
        if filter_group and "OK_KO_Label" in df.columns:
            plot_df = df[df["OK_KO_Label"] == filter_group].copy()
            group_label = f" ({filter_group} samples only)"

        res = self.plotter.plot_distribution_comparison(plot_df, column=column, plot_type=plot_type)

        if not res.get("success", False):
            return {"success": False, "message": f"❌ {res.get('error', 'Distribution plot failed')}"}

        return {
            "success": True,
            "message": f"✅ Generated **{plot_type}** distribution plot for **{column}**{group_label}",
            "plot": res["figure"],
            "summary": res.get("summary", {"plot_type": f"distribution_{plot_type}", "column": column}),
        }

    def _tool_compare_features(self, columns: List[str]) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None
        res = self.plotter.plot_feature_comparison(df, columns=columns)

        if not res.get("success", False):
            return {"success": False, "message": f"❌ {res.get('error', 'Comparison plot failed')}"}

        return {
            "success": True,
            "message": f"✅ Generated feature comparison plot for: {', '.join(res.get('columns', columns))}",
            "plot": res["figure"],
            "summary": res.get("summary", {"plot_type": "feature_comparison", "column": ",".join(columns)}),
        }

    def _tool_get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        try:
            ranking = []
            debug_info = []  # For debugging

            # Try multiple paths to find feature importance data
            if self.analysis_results:
                debug_info.append(f"Available keys: {list(self.analysis_results.keys())}")
                
                # Path 1: Top-level feature_ranking (added by advanced_analysis.py)
                if "feature_ranking" in self.analysis_results:
                    feature_ranking = self.analysis_results.get("feature_ranking", [])
                    debug_info.append(f"Path 1 - feature_ranking length: {len(feature_ranking) if feature_ranking else 0}")
                    if feature_ranking and len(feature_ranking) > 0:
                        debug_info.append(f"Path 1 - First item keys: {list(feature_ranking[0].keys())}")
                        for idx, item in enumerate(feature_ranking[:top_n], start=1):
                            # Statistical analysis uses 'composite_score', ML uses 'importance'
                            importance_value = item.get("importance") or item.get("composite_score", 0)
                            ranking.append(
                                {
                                    "rank": idx,
                                    "feature": str(item.get("feature", "")),
                                    "importance": float(importance_value),
                                }
                            )
                
                # Path 2: ML feature importance results
                if not ranking and "ml_feature_importance" in self.analysis_results:
                    ml_fi = self.analysis_results.get("ml_feature_importance", {})
                    debug_info.append(f"Path 2 - ml_feature_importance keys: {list(ml_fi.keys()) if ml_fi else 'None'}")
                    if ml_fi and "feature_importance" in ml_fi:
                        feature_ranking = ml_fi["feature_importance"].get("feature_ranking", [])
                        debug_info.append(f"Path 2 - feature_ranking length: {len(feature_ranking) if feature_ranking else 0}")
                        if feature_ranking and len(feature_ranking) > 0:
                            debug_info.append(f"Path 2 - First item keys: {list(feature_ranking[0].keys())}")
                            for idx, item in enumerate(feature_ranking[:top_n], start=1):
                                ranking.append(
                                    {
                                        "rank": idx,
                                        "feature": str(item.get("feature", "")),
                                        "importance": float(item.get("importance", 0)),
                                    }
                                )
                
                # Path 3: Legacy path (for backward compatibility)
                if not ranking and "feature_importance" in self.analysis_results:
                    fi = self.analysis_results.get("feature_importance", {})
                    feature_ranking = fi.get("feature_importance", {}).get("feature_ranking", [])
                    if feature_ranking:
                        for idx, item in enumerate(feature_ranking[:top_n], start=1):
                            importance_value = item.get("importance") or item.get("composite_score", 0)
                            ranking.append(
                                {
                                    "rank": idx,
                                    "feature": str(item.get("feature", "")),
                                    "importance": float(importance_value),
                                }
                            )

            if not ranking:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                csv_path = os.path.join(base_dir, "data", "processed", "feature_importance.csv")

                if not os.path.exists(csv_path):
                    return {"success": False, "message": "⚠️ No feature importance data available. Please run Advanced Analysis first."}

                try:
                    imp = pd.read_csv(csv_path)
                except Exception:
                    imp = pd.read_csv(csv_path, header=None)

                if set(["feature", "importance"]).issubset(set(imp.columns)):
                    df2 = imp[["feature", "importance"]].copy()
                else:
                    df2 = imp.iloc[:, :2].copy()
                    df2.columns = ["c0", "c1"]
                    c0_num = pd.to_numeric(df2["c0"], errors="coerce").notna().mean()
                    c1_num = pd.to_numeric(df2["c1"], errors="coerce").notna().mean()
                    if c0_num < c1_num:
                        df2 = df2.rename(columns={"c0": "feature", "c1": "importance"})
                    else:
                        df2 = df2.rename(columns={"c1": "feature", "c0": "importance"})

                df2["feature"] = df2["feature"].astype(str)
                df2["importance"] = pd.to_numeric(df2["importance"], errors="coerce")
                df2 = df2.dropna(subset=["importance"])
                df2 = df2.sort_values("importance", ascending=False).reset_index(drop=True)

                for rank_i, row in enumerate(df2.head(top_n).itertuples(index=False), start=1):
                    ranking.append({"rank": rank_i, "feature": str(row.feature), "importance": float(row.importance)})

            if not ranking:
                return {"success": False, "message": f"⚠️ No feature importance data available. Please run Advanced Analysis first."}

            # Get global task context to include in the result
            task_context = self._get_global_task_context()
            
            msg = ["🎯 **Top Important Features**\n"]
            for r in ranking:
                msg.append(f"{r['rank']}. **{r['feature']}** — {r['importance']:.6f}")
            
            # Return with task context embedded
            return {
                "success": True, 
                "message": "\n".join(msg).strip(), 
                "data": {
                    "feature_importance": ranking,
                    "task_context": task_context  # CRITICAL: Include this for LLM
                },
                "summary": {
                    "plot_type": "feature_importance",
                    "top_n": len(ranking),
                    "has_groups": task_context.get("label_column") is not None,
                    "group_column": task_context.get("label_column"),
                    "groups": task_context.get("groups", []),
                    "analysis_goal": task_context.get("analysis_goal"),
                    "note": f"Feature importance ranking based on ability to distinguish between groups."
                }
            }

        except Exception as e:
            return {"success": False, "message": f"⚠️ Error reading feature importance: {str(e)}"}
