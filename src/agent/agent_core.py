"""
Statistical AI Agent Core Module (Stable Version)
- Deterministic tools produce ALL numbers and plot summaries.
- Agent NEVER asks LLM to compute or invent numeric facts.
- LLM is only used for fallback general chat, not for tool-result generation.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Callable, Tuple

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

        # If intent unknown -> optional LLM fallback (chat only, no numbers)
        if intent["type"] == "unknown":
            if self.enable_llm_fallback_chat:
                return self._fallback_chat(user_message)
            return {
                "response": "⚠️ I couldn't understand the request. Try: 'mean and variance of sensor_2', or 'plot histogram of sensor_11'.",
                "plots": [],
                "tool_calls": None,
                "tool_results": [],
            }

        # Execute deterministic tool
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

        # Hard stop: never ask LLM to “fix” a tool failure
        if not result.get("success", False):
            resp = result.get("message") or f"❌ Tool error: {result.get('error', 'Unknown error')}"
            self.conversation.add_message("assistant", resp, {"tool_results": [result]})
            return {
                "response": resp,
                "plots": [],
                "tool_calls": None,
                "tool_results": [result],
            }

        # Build final response from tool outputs (stable)
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

    def _extract_metrics(self, text: str) -> Optional[List[str]]:
        """Return a list of requested metrics or None meaning 'full summary'."""
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

        # If user says summary/statistics and no explicit metrics -> full summary
        if any(k in text for k in ["summary", "statistics", "statistical summary"]) and not requested:
            return None

        return sorted(set(requested)) if requested else None

    def _parse_intent(self, user_message: str) -> Dict[str, Any]:
        text = user_message.lower()
        cols = self._match_columns(user_message)

        # Feature importance (explicit)
        if any(k in text for k in ["feature importance", "importance ranking", "rank features", "most important features"]):
            return {"type": "tool", "tool": "get_feature_importance", "args": {"top_n": 10}}

        # FFT / frequency spectrum - support multiple ways to ask
        # "fft", "frequency spectrum", "show fft for X", "plot frequency of X"
        fft_keywords = ["fft", "frequency spectrum", "fourier", "spectrum"]
        if any(k in text for k in fft_keywords):
            if not cols:
                return {"type": "unknown"}
            return {"type": "tool", "tool": "plot_frequency_spectrum", "args": {"column": cols[0]}}

        # Detect if user wants to filter by OK or KO group
        filter_group = None
        if " ok " in f" {text} " or "ok samples" in text or "ok group" in text or "for ok" in text:
            filter_group = "OK"
        elif " ko " in f" {text} " or "ko samples" in text or "ko group" in text or "for ko" in text:
            filter_group = "KO"

        # Time series (explicit phrase only)
        if "time series" in text or "timeseries" in text:
            if not cols:
                return {"type": "unknown"}
            return {"type": "tool", "tool": "plot_time_series", "args": {"column": cols[0], "separate_groups": True, "filter_group": filter_group}}

        # Distribution plot: require explicit plot-type words (NOT just "show")
        if any(k in text for k in ["histogram", "distribution", "boxplot", "box plot", "violin", "kde", "density"]):
            if not cols:
                return {"type": "unknown"}
            plot_type = "histogram"
            if "boxplot" in text or "box plot" in text:
                plot_type = "boxplot"
            elif "violin" in text:
                plot_type = "violin"
            elif "kde" in text or "density" in text:
                plot_type = "kde"
            return {"type": "tool", "tool": "plot_distribution", "args": {"column": cols[0], "plot_type": plot_type, "filter_group": filter_group}}

        # Statistics: detect numeric words
        stat_words = ["mean", "median", "mode", "variance", "std", "standard deviation", "summary", "statistics", "min", "max", "count", "average"]
        if any(w in text for w in stat_words):
            metrics = self._extract_metrics(text)
            return {
                "type": "tool",
                "tool": "get_statistical_summary",
                "args": {"columns": cols if cols else None, "group_by_ok_ko": True, "metrics": metrics},
            }

        # Compare multiple features (explicit)
        if "compare" in text and len(cols) >= 2:
            return {"type": "tool", "tool": "compare_features", "args": {"columns": cols[:6]}}

        return {"type": "unknown"}

    # -----------------------------
    # Rendering (NO LLM NUMBERS)
    # -----------------------------
    def _render_response_from_tool(self, tool_result: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """
        Construct final response ONLY using fields returned by the tool:
        - tool_result['message'] (human-readable)
        - tool_result['data'] (structured)
        - tool_result['summary'] (structured explanation for plots)
        
        Optionally uses LLM to interpret the results if enable_llm_interpretation is True.
        """
        parts = []

        # Always include tool message (already deterministic)
        if tool_result.get("message"):
            parts.append(tool_result["message"].strip())

        # If plot tool returned summary -> generate stable explanation
        summary = tool_result.get("summary")
        if isinstance(summary, dict) and summary:
            parts.append("\n---\n")
            parts.append(self._explain_plot_summary(summary))

        # Include warnings if any
        if tool_result.get("warning"):
            parts.append(f"\n⚠️ {tool_result['warning']}")
        
        # Optionally add LLM interpretation
        if self.enable_llm_interpretation:
            base_response = "\n".join([p for p in parts if p]).strip()
            llm_interpretation = self._llm_interpret_result(tool_result, intent, base_response)
            if llm_interpretation:
                parts.append("\n---\n")
                parts.append("🤖 **AI Analysis:**")
                parts.append(llm_interpretation)

        return "\n".join([p for p in parts if p]).strip()

    def _explain_plot_summary(self, summary: Dict[str, Any]) -> str:
        """
        Template explanation. No invented numbers.
        Expect plotter to provide summary fields.
        """
        plot_type = summary.get("plot_type", "plot")
        column = summary.get("column", "feature")

        lines = [f"🧾 **Plot Interpretation (from tool summary)**", f"- Plot type: **{plot_type}**", f"- Column: **{column}**"]

        # Optional keys depending on plot type
        if "group_stats" in summary:
            # e.g., {"OK": {"mean":..., "std":...}, "KO": {...}}
            gs = summary["group_stats"]
            for g, st in gs.items():
                lines.append(f"- {g} stats: {st}")

        if "dominant_frequencies" in summary:
            # e.g., {"OK": [(freq, magnitude), ...], "KO": ...}
            dfreq = summary["dominant_frequencies"]
            lines.append("- Dominant frequencies (top peaks):")
            for g, peaks in dfreq.items():
                lines.append(f"  - {g}: {peaks}")

        if summary.get("note"):
            lines.append(f"- Note: {summary['note']}")

        return "\n".join(lines)

    def _llm_interpret_result(self, tool_result: Dict[str, Any], intent: Dict[str, Any], base_response: str) -> str:
        """
        Use LLM to provide intelligent interpretation of tool results.
        The LLM receives the exact numbers from tools and explains their meaning.
        """
        try:
            tool_name = intent.get("tool", "unknown")
            
            # Build context for LLM
            context_parts = []
            context_parts.append(f"Tool used: {tool_name}")
            context_parts.append(f"Tool output:\n{base_response}")
            
            # Add structured data if available
            if tool_result.get("data"):
                context_parts.append(f"Structured data: {tool_result['data']}")
            
            context = "\n".join(context_parts)
            
            # Domain-specific interpretation prompt
            interpretation_prompt = f"""You are an expert data analyst interpreting results from an industrial sensor analysis tool.

The data is from NASA C-MAPSS turbofan engine degradation dataset:
- OK = healthy engine (RUL > threshold)
- KO = degraded engine (RUL <= threshold, approaching failure)
- Sensors measure temperature, pressure, speed, etc.

Here are the EXACT results from the analysis tool (DO NOT change any numbers):

{context}

Provide a brief (2-3 sentences) expert interpretation:
1. What do these numbers mean in the context of engine health?
2. Is there a significant difference between OK and KO groups?
3. What actionable insight can be drawn?

IMPORTANT: Only explain the numbers shown above. Do NOT invent new statistics."""

            messages = [
                {"role": "system", "content": "You are a predictive maintenance expert. Interpret analysis results concisely."},
                {"role": "user", "content": interpretation_prompt}
            ]
            
            llm_response = self.llm.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent interpretation
                max_tokens=200,
                tools=None
            )
            
            interpretation = (llm_response.get("content") or "").strip()
            return interpretation if interpretation else None
            
        except Exception as e:
            # Silent fail - interpretation is optional
            return None

    # -----------------------------
    # LLM fallback chat (no numbers)
    # -----------------------------
    def _fallback_chat(self, user_message: str) -> Dict[str, Any]:
        messages = self.conversation.get_messages_for_llm()

        # HARD safety prompt: forbid numerical claims
        system_prompt = (
            "You are a helpful assistant for analyzing industrial sensor data (like NASA C-MAPSS turbofan engine data). "
            "Do NOT compute or invent any numeric results. "
            "If user asks for statistics or plots, instruct them how to ask using tool keywords "
            "like: 'mean variance sensor_2', 'histogram sensor_11', 'fft sensor_7', 'time series for KO samples'."
        )
        messages.append({"role": "system", "content": system_prompt})

        llm_response = self.llm.generate(messages=messages, temperature=0.6, max_tokens=500, tools=None)
        content = (llm_response.get("content") or "").strip()

        if not content:
            content = "I couldn't parse that. Try: 'mean and variance of sensor_2', or 'plot histogram of sensor_11'."

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
                metrics = None  # full summary

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

        return {"success": True, "message": "\n".join(msg).strip(), "data": summary}

    def _tool_plot_time_series(self, column: str, separate_groups: bool = True, filter_group: str = None) -> Dict[str, Any]:
        df = self.current_data
        assert df is not None
        
        # Filter by group if specified
        plot_df = df
        group_label = ""
        if filter_group and "OK_KO_Label" in df.columns:
            plot_df = df[df["OK_KO_Label"] == filter_group].copy()
            group_label = f" ({filter_group} samples only)"
            separate_groups = False  # No need to separate when already filtered
        
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
        
        # Filter by group if specified
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
        """
        Get feature importance from current analysis results or CSV file.
        Prioritizes current session analysis results over saved CSV.
        """
        try:
            ranking = []
            
            # Priority 1: Use current analysis results (from this session)
            if self.analysis_results and 'feature_importance' in self.analysis_results:
                fi = self.analysis_results.get('feature_importance', {})
                feature_ranking = fi.get('feature_importance', {}).get('feature_ranking', [])
                if feature_ranking:
                    for item in feature_ranking[:top_n]:
                        ranking.append({
                            "rank": item.get('rank', len(ranking) + 1),
                            "feature": str(item.get('feature', '')),
                            "importance": float(item.get('importance', 0))
                        })
            
            # Priority 2: Fall back to CSV file if no current results
            if not ranking:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                csv_path = os.path.join(base_dir, "data", "processed", "feature_importance.csv")

                if not os.path.exists(csv_path):
                    return {"success": False, "message": "⚠️ No feature importance data available. Please run Advanced Analysis first."}

                # Read CSV
                try:
                    imp = pd.read_csv(csv_path)
                except Exception:
                    imp = pd.read_csv(csv_path, header=None)

                # Normalize columns
                if set(["feature", "importance"]).issubset(set(imp.columns)):
                    df = imp[["feature", "importance"]].copy()
                else:
                    df = imp.iloc[:, :2].copy()
                    df.columns = ["c0", "c1"]
                    c0_num = pd.to_numeric(df["c0"], errors="coerce").notna().mean()
                    c1_num = pd.to_numeric(df["c1"], errors="coerce").notna().mean()
                    if c0_num < c1_num:
                        df = df.rename(columns={"c0": "feature", "c1": "importance"})
                    else:
                        df = df.rename(columns={"c1": "feature", "c0": "importance"})

                df["feature"] = df["feature"].astype(str)
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
                df = df.dropna(subset=["importance"])
                df = df.sort_values("importance", ascending=False).reset_index(drop=True)

                for rank_i, row in enumerate(df.head(top_n).itertuples(index=False), start=1):
                    ranking.append({"rank": rank_i, "feature": str(row.feature), "importance": float(row.importance)})

            if not ranking:
                return {"success": False, "message": "⚠️ No valid feature importance rows found."}

            msg = ["🎯 **Top Important Features**\n"]
            for r in ranking:
                msg.append(f"{r['rank']}. **{r['feature']}** — {r['importance']:.6f}")
            return {"success": True, "message": "\n".join(msg).strip(), "data": {"feature_importance": ranking}}

        except Exception as e:
            return {"success": False, "message": f"⚠️ Error reading feature importance: {str(e)}"}
