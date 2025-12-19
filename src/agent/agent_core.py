"""
Statistical AI Agent Core Module
Orchestrates LLM, tools, and conversation management
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from .llm_interface import LLMInterface
from .conversation import ConversationManager
from .plotting_tools import PlottingTools


class StatisticalAgent:
    """
    AI Agent for statistical analysis and visualization
    Uses LLM to understand user requests and call appropriate tools
    """
    
    def __init__(self, 
                 llm_backend: str = "ollama",
                 llm_model: str = None,
                 api_key: str = None):
        """
        Initialize the Statistical AI Agent
        
        Args:
            llm_backend: "ollama" or "openai"
            llm_model: Model name
            api_key: API key for OpenAI (if using openai backend)
        """
        self.llm = LLMInterface(backend=llm_backend, model=llm_model, api_key=api_key)
        self.conversation = ConversationManager()
        self.plotter = PlottingTools()
        
        # Current data context
        self.current_data: Optional[pd.DataFrame] = None
        self.data_info: Dict[str, Any] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # Register available tools
        self.tools = self._register_tools()
        self.tool_functions = self._register_tool_functions()
    
    def set_data_context(self, df: pd.DataFrame, data_info: Dict = None):
        """
        Set the current dataset context for analysis
        
        Args:
            df: DataFrame to analyze
            data_info: Additional information about the dataset
        """
        self.current_data = df
        self.data_info = data_info or {}
        
        # Update conversation context
        context = self._create_data_context_summary()
        self.conversation.add_context(context)
    
    def set_analysis_results(self, results: Dict[str, Any]):
        """Set statistical analysis results"""
        self.analysis_results = results
    
    def _create_data_context_summary(self) -> str:
        """Create a summary of current data context"""
        if self.current_data is None:
            return "No data loaded yet."
        
        df = self.current_data
        summary_parts = [
            f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.",
            f"\nColumns: {', '.join(df.columns.tolist())}",
        ]
        
        # Add OK/KO distribution if available
        if 'OK_KO_Label' in df.columns:
            ok_count = (df['OK_KO_Label'] == 'OK').sum()
            ko_count = (df['OK_KO_Label'] == 'KO').sum()
            summary_parts.append(f"\nOK samples: {ok_count}, KO samples: {ko_count}")
        
        # Add numerical columns info
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            summary_parts.append(f"\nNumerical features: {', '.join(numerical_cols)}")
        
        return "\n".join(summary_parts)
    
    def chat(self, user_message: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process user message and generate response
        
        Args:
            user_message: User's message/request
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response and any generated plots
        """
        if self.current_data is None:
            return {
                'response': "⚠️ Please load a dataset first before asking questions.",
                'plots': [],
                'tool_calls': None
            }
        
        # Add user message to conversation
        self.conversation.add_message('user', user_message)
        
        # First, try to detect intent and call tools directly (more reliable for Ollama)
        plots = []
        tool_results = []
        direct_tool_result = self._detect_and_execute_tools(user_message)
        
        if direct_tool_result:
            tool_results = direct_tool_result.get('results', [])
            plots = direct_tool_result.get('plots', [])
            
            # Generate natural language response about the tool execution
            response_content = direct_tool_result.get('summary', '')
        else:
            # Fall back to LLM for general questions
            messages = self.conversation.get_messages_for_llm()
            llm_response = self.llm.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                tools=None  # Don't use tool calling for Ollama
            )
            
            # Check for errors
            if 'error' in llm_response:
                error_msg = llm_response.get('content', 'Unknown error occurred')
                self.conversation.add_message('assistant', error_msg)
                return {
                    'response': error_msg,
                    'plots': [],
                    'tool_calls': None,
                    'error': llm_response['error']
                }
            
            response_content = llm_response.get('content', '')
        
        # Add assistant response to conversation
        self.conversation.add_message('assistant', response_content, {
            'tool_results': tool_results
        })
        
        return {
            'response': response_content,
            'plots': plots,
            'tool_calls': None,
            'tool_results': tool_results
        }
    
    def _detect_and_execute_tools(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Detect intent from user message and execute appropriate tools directly"""
        message_lower = user_message.lower()
        plots = []
        results = []
        summary_parts = []
        
        # Extract column names from message
        columns = [col for col in self.current_data.columns if col.lower() in message_lower and col != 'OK_KO_Label']
        
        # Detect statistical summary requests (check first to prioritize over plotting)
        stat_keywords = ['statistic', 'summary', 'mean', 'median', 'standard deviation', 
                         'average', 'variance', 'std', 'deviation', 'count', 'max', 'min']
        if any(word in message_lower for word in stat_keywords):
            # Check if asking for specific column statistics
            result = self._tool_get_statistical_summary(columns if columns else None, group_by_ok_ko=True)
            results.append(result)
            summary_parts.append(result.get('message', 'Statistical summary generated'))
        
        # Detect plot requests
        elif any(word in message_lower for word in ['plot', 'show', 'display', 'visualize', 'draw', 'compare', 'distribution']):
            if columns:
                # Determine plot type
                plot_type = 'histogram'  # default
                if 'boxplot' in message_lower or 'box plot' in message_lower:
                    plot_type = 'boxplot'
                elif 'violin' in message_lower:
                    plot_type = 'violin'
                elif 'kde' in message_lower or 'density' in message_lower:
                    plot_type = 'kde'
                
                result = self._tool_plot_distribution(columns[0], plot_type=plot_type)
                results.append(result)
                if result.get('plot'):
                    plots.append(result['plot'])
                    summary_parts.append(f"📊 Distribution comparison for **{columns[0]}** between OK and KO groups")
            
            # Multiple feature comparison
            elif 'compare' in message_lower and len(columns) > 1:
                result = self._tool_compare_features(columns)
                results.append(result)
                if result.get('plot'):
                    plots.append(result['plot'])
                    summary_parts.append(f"📊 Comparison of {len(columns)} features")
        
        # Detect feature importance requests
        elif any(word in message_lower for word in ['important', 'feature importance', 'ranking', 'top feature', 'which feature']):
            result = self._tool_get_feature_importance(top_n=10)
            results.append(result)
            summary_parts.append(result.get('message', 'Feature importance retrieved'))
        
        if results:
            return {
                'results': results,
                'plots': plots,
                'summary': '\n\n'.join(summary_parts) if summary_parts else 'Analysis complete'
            }
        
        return None
    
    def _execute_tool_call(self, tool_call: Dict) -> Dict[str, Any]:
        """Execute a tool function call"""
        try:
            if tool_call['type'] == 'function':
                func_name = tool_call['function']['name']
                func_args_str = tool_call['function']['arguments']
                
                # Parse arguments
                if isinstance(func_args_str, str):
                    func_args = json.loads(func_args_str)
                else:
                    func_args = func_args_str
                
                # Get the tool function
                if func_name not in self.tool_functions:
                    return {
                        'success': False,
                        'message': f"Tool '{func_name}' not found",
                        'error': 'unknown_tool'
                    }
                
                # Execute the function
                tool_func = self.tool_functions[func_name]
                result = tool_func(**func_args)
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error executing tool: {str(e)}",
                'error': str(e)
            }
    
    def _register_tools(self) -> List[Dict]:
        """Register available tools for LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_statistical_summary",
                    "description": "Get statistical summary (mean, median, mode, std, variance) for features. Compares OK vs KO groups.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to analyze. If empty, analyze all numerical columns."
                            },
                            "group_by_ok_ko": {
                                "type": "boolean",
                                "description": "Whether to split analysis by OK/KO groups"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_distribution",
                    "description": "Plot distribution comparison between OK and KO groups. Supports histogram, boxplot, violin plot, and KDE.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "description": "Column name to plot"
                            },
                            "plot_type": {
                                "type": "string",
                                "enum": ["histogram", "kde", "boxplot", "violin"],
                                "description": "Type of distribution plot"
                            }
                        },
                        "required": ["column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_feature_importance",
                    "description": "Get feature importance ranking showing which features best discriminate between OK and KO.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of top features to return (default 10)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_features",
                    "description": "Compare multiple features side by side between OK and KO groups.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to compare"
                            }
                        },
                        "required": ["columns"]
                    }
                }
            }
        ]
    
    def _register_tool_functions(self) -> Dict[str, Callable]:
        """Map tool names to actual Python functions"""
        return {
            'get_statistical_summary': self._tool_get_statistical_summary,
            'plot_time_series': self._tool_plot_time_series,
            'plot_frequency_spectrum': self._tool_plot_frequency_spectrum,
            'plot_distribution': self._tool_plot_distribution,
            'get_feature_importance': self._tool_get_feature_importance,
            'compare_features': self._tool_compare_features
        }
    
    # Tool function implementations
    
    def _tool_get_statistical_summary(self, columns: List[str] = None, 
                                     group_by_ok_ko: bool = True) -> Dict[str, Any]:
        """Get statistical summary"""
        df = self.current_data
        
        if columns is None or len(columns) == 0:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove OK_KO_Label from analysis
        columns = [c for c in columns if c in df.columns and c != 'OK_KO_Label']
        
        if not columns:
            return {'success': False, 'message': 'No valid numerical columns found'}
        
        summary = {}
        
        if group_by_ok_ko and 'OK_KO_Label' in df.columns:
            for col in columns:
                ok_data = df[df['OK_KO_Label'] == 'OK'][col].dropna()
                ko_data = df[df['OK_KO_Label'] == 'KO'][col].dropna()
                
                summary[col] = {
                    'OK': {
                        'mean': float(ok_data.mean()),
                        'median': float(ok_data.median()),
                        'std': float(ok_data.std()),
                        'variance': float(ok_data.var()),
                        'min': float(ok_data.min()),
                        'max': float(ok_data.max())
                    },
                    'KO': {
                        'mean': float(ko_data.mean()),
                        'median': float(ko_data.median()),
                        'std': float(ko_data.std()),
                        'variance': float(ko_data.var()),
                        'min': float(ko_data.min()),
                        'max': float(ko_data.max())
                    }
                }
        else:
            for col in columns:
                data = df[col].dropna()
                summary[col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
        
        # Format message
        message = "📊 Statistical Summary:\n\n"
        for col, stats in summary.items():
            message += f"**{col}:**\n"
            if 'OK' in stats:
                message += f"  OK - Mean: {stats['OK']['mean']:.3f}, Median: {stats['OK']['median']:.3f}, Std: {stats['OK']['std']:.3f}\n"
                message += f"  KO - Mean: {stats['KO']['mean']:.3f}, Median: {stats['KO']['median']:.3f}, Std: {stats['KO']['std']:.3f}\n"
            else:
                message += f"  Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}, Std: {stats['std']:.3f}\n"
        
        return {
            'success': True,
            'message': message,
            'data': summary
        }
    
    def _tool_plot_time_series(self, column: str, separate_groups: bool = True) -> Dict[str, Any]:
        """Plot time series"""
        result = self.plotter.plot_time_series(
            self.current_data,
            column=column,
            separate_groups=separate_groups
        )
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"✅ Generated time series plot for {column}",
                'plot': result['figure']
            }
        else:
            return {
                'success': False,
                'message': f"❌ Error: {result.get('error')}"
            }
    
    def _tool_plot_frequency_spectrum(self, column: str, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """Plot frequency spectrum"""
        result = self.plotter.plot_frequency_spectrum(
            self.current_data,
            column=column,
            sampling_rate=sampling_rate
        )
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"✅ Generated frequency spectrum plot for {column}",
                'plot': result['figure']
            }
        else:
            return {
                'success': False,
                'message': f"❌ Error: {result.get('error')}"
            }
    
    def _tool_plot_distribution(self, column: str, plot_type: str = 'histogram') -> Dict[str, Any]:
        """Plot distribution comparison"""
        result = self.plotter.plot_distribution_comparison(
            self.current_data,
            column=column,
            plot_type=plot_type
        )
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"✅ Generated distribution plot for {column}",
                'plot': result['figure']
            }
        else:
            return {
                'success': False,
                'message': f"❌ Error: {result.get('error')}"
            }
    
    def _tool_get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """Get feature importance from analysis results"""
        if not self.analysis_results or 'feature_importance' not in self.analysis_results:
            return {
                'success': False,
                'message': '⚠️ Feature importance analysis not available. Please run feature importance analysis first.'
            }
        
        ranking = self.analysis_results['feature_importance']['feature_ranking'][:top_n]
        
        message = f"🎯 Top {len(ranking)} Important Features:\n\n"
        for item in ranking:
            message += f"{item['rank']}. **{item['feature']}** - Importance: {item['importance']:.4f}\n"
        
        return {
            'success': True,
            'message': message,
            'data': ranking
        }
    
    def _tool_compare_features(self, columns: List[str]) -> Dict[str, Any]:
        """Compare multiple features"""
        result = self.plotter.plot_feature_comparison(
            self.current_data,
            columns=columns
        )
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"✅ Generated comparison plot for {len(columns)} features",
                'plot': result['figure']
            }
        else:
            return {
                'success': False,
                'message': f"❌ Error: {result.get('error')}"
            }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation.clear_history()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return self.conversation.get_full_history()
