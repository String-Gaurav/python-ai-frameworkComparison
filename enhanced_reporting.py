"""
Enhanced Reporting System with Charts and Visualizations
Generates rich HTML reports with interactive charts using Plotly
"""

import json
import datetime
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)


class EnhancedReportGenerator:
    """Generate rich HTML reports with charts and detailed analytics"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or config.get("output.directory", "test_results"))
        self.output_dir.mkdir(exist_ok=True)
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - charts will be disabled")
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available - advanced analytics will be limited")
    
    def generate_comprehensive_report(
        self, 
        results: Dict[str, List[Any]], 
        title: str = "LLM Comparison Report"
    ) -> str:
        """Generate a comprehensive HTML report with charts and analytics"""
        
        report_data = self._process_results(results)
        
        html_content = self._generate_html_report(report_data, title)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"enhanced_report_{timestamp}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Enhanced report generated: {report_path}")
        return str(report_path)
    
    def _process_results(self, results: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Process raw results into structured data for reporting"""
        processed_data = {
            "models": {},
            "summary": {},
            "categories": {},
            "performance_trends": [],
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "total_models": len(results),
                "total_tests": sum(len(model_results) for model_results in results.values())
            }
        }
        
        all_response_times = []
        all_quality_scores = []
        category_performance = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Extract metrics
            response_times = [r.response_time for r in model_results if hasattr(r, 'response_time')]
            quality_scores = [r.scores.get('overall_quality', 0) for r in model_results if hasattr(r, 'scores')]
            
            # Model-specific analytics
            model_data = {
                "name": model_name,
                "test_count": len(model_results),
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "avg_quality": statistics.mean(quality_scores) if quality_scores else 0,
                "min_quality": min(quality_scores) if quality_scores else 0,
                "max_quality": max(quality_scores) if quality_scores else 0,
                "std_quality": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                "response_times": response_times,
                "quality_scores": quality_scores
            }
            
            # Category breakdown
            model_categories = {}
            for result in model_results:
                if hasattr(result, 'test_id'):
                    # Extract category from test_id (assuming format like "category_testname")
                    category = result.test_id.split('_')[0] if '_' in result.test_id else 'general'
                    if category not in model_categories:
                        model_categories[category] = []
                    
                    model_categories[category].append({
                        'response_time': result.response_time,
                        'quality': result.scores.get('overall_quality', 0)
                    })
            
            model_data["categories"] = model_categories
            processed_data["models"][model_name] = model_data
            
            # Aggregate data
            all_response_times.extend(response_times)
            all_quality_scores.extend(quality_scores)
            
            # Category performance tracking
            for category, cat_results in model_categories.items():
                if category not in category_performance:
                    category_performance[category] = {}
                category_performance[category][model_name] = {
                    'avg_time': statistics.mean([r['response_time'] for r in cat_results]),
                    'avg_quality': statistics.mean([r['quality'] for r in cat_results])
                }
        
        # Summary statistics
        processed_data["summary"] = {
            "avg_response_time": statistics.mean(all_response_times) if all_response_times else 0,
            "avg_quality": statistics.mean(all_quality_scores) if all_quality_scores else 0,
            "best_response_time_model": min(processed_data["models"].keys(), 
                                          key=lambda x: processed_data["models"][x]["avg_response_time"]) if processed_data["models"] else None,
            "best_quality_model": max(processed_data["models"].keys(), 
                                    key=lambda x: processed_data["models"][x]["avg_quality"]) if processed_data["models"] else None
        }
        
        processed_data["categories"] = category_performance
        
        return processed_data
    
    def _generate_html_report(self, data: Dict[str, Any], title: str) -> str:
        """Generate HTML report with embedded charts"""
        
        # Generate charts
        charts_html = ""
        if PLOTLY_AVAILABLE:
            charts_html = self._generate_charts(data)
        
        # Create HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e9ecef;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    text-transform: uppercase;
                    opacity: 0.9;
                }}
                .summary-card .value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 0;
                }}
                .chart-section {{
                    margin: 40px 0;
                }}
                .chart-title {{
                    font-size: 20px;
                    font-weight: bold;
                    margin-bottom: 20px;
                    color: #333;
                }}
                .model-details {{
                    margin-top: 40px;
                }}
                .model-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric {{
                    text-align: center;
                    padding: 10px;
                    background: white;
                    border-radius: 5px;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #e9ecef;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on {data['metadata']['generated_at']}</p>
                </div>
                
                {self._generate_summary_section(data)}
                
                <div class="chart-section">
                    {charts_html}
                </div>
                
                {self._generate_model_details_section(data)}
                
                <div class="footer">
                    <p>Report generated by LLM Test Framework</p>
                    <p>Total Models: {data['metadata']['total_models']} | Total Tests: {data['metadata']['total_tests']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_summary_section(self, data: Dict[str, Any]) -> str:
        """Generate the summary cards section"""
        summary = data["summary"]
        
        return f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Average Response Time</h3>
                <p class="value">{summary['avg_response_time']:.2f}s</p>
            </div>
            <div class="summary-card">
                <h3>Average Quality Score</h3>
                <p class="value">{summary['avg_quality']:.2f}</p>
            </div>
            <div class="summary-card">
                <h3>Fastest Model</h3>
                <p class="value">{summary.get('best_response_time_model', 'N/A')}</p>
            </div>
            <div class="summary-card">
                <h3>Highest Quality</h3>
                <p class="value">{summary.get('best_quality_model', 'N/A')}</p>
            </div>
        </div>
        """
    
    def _generate_model_details_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed model information"""
        html = '<div class="model-details"><h2>Model Details</h2>'
        
        for model_name, model_data in data["models"].items():
            html += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Tests Run</div>
                        <div class="metric-value">{model_data['test_count']}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Response Time</div>
                        <div class="metric-value">{model_data['avg_response_time']:.2f}s</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Quality</div>
                        <div class="metric-value">{model_data['avg_quality']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Time Std Dev</div>
                        <div class="metric-value">{model_data['std_response_time']:.2f}s</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Quality Std Dev</div>
                        <div class="metric-value">{model_data['std_quality']:.2f}</div>
                    </div>
                </div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_charts(self, data: Dict[str, Any]) -> str:
        """Generate Plotly charts as HTML"""
        if not PLOTLY_AVAILABLE:
            return "<p>Charts unavailable - Plotly not installed</p>"
        
        charts_html = ""
        
        # Response Time Comparison Chart
        fig1 = self._create_response_time_chart(data)
        if fig1:
            charts_html += f"""
            <div class="chart-title">Response Time Comparison</div>
            <div id="response-time-chart"></div>
            <script>
                {fig1.to_html(div_id="response-time-chart", include_plotlyjs=False)}
            </script>
            """
        
        # Quality Score Comparison Chart
        fig2 = self._create_quality_chart(data)
        if fig2:
            charts_html += f"""
            <div class="chart-title">Quality Score Comparison</div>
            <div id="quality-chart"></div>
            <script>
                {fig2.to_html(div_id="quality-chart", include_plotlyjs=False)}
            </script>
            """
        
        # Performance Scatter Plot
        fig3 = self._create_performance_scatter(data)
        if fig3:
            charts_html += f"""
            <div class="chart-title">Performance Overview (Speed vs Quality)</div>
            <div id="performance-scatter"></div>
            <script>
                {fig3.to_html(div_id="performance-scatter", include_plotlyjs=False)}
            </script>
            """
        
        return charts_html
    
    def _create_response_time_chart(self, data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create response time comparison bar chart"""
        if not data["models"]:
            return None
        
        models = list(data["models"].keys())
        avg_times = [data["models"][model]["avg_response_time"] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=avg_times,
                marker_color='skyblue',
                text=[f'{time:.2f}s' for time in avg_times],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Average Response Time by Model",
            xaxis_title="Model",
            yaxis_title="Response Time (seconds)",
            height=400
        )
        
        return fig
    
    def _create_quality_chart(self, data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create quality score comparison chart"""
        if not data["models"]:
            return None
        
        models = list(data["models"].keys())
        avg_quality = [data["models"][model]["avg_quality"] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=avg_quality,
                marker_color='lightcoral',
                text=[f'{quality:.2f}' for quality in avg_quality],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Average Quality Score by Model",
            xaxis_title="Model",
            yaxis_title="Quality Score",
            height=400
        )
        
        return fig
    
    def _create_performance_scatter(self, data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create performance scatter plot (response time vs quality)"""
        if not data["models"]:
            return None
        
        models = []
        response_times = []
        quality_scores = []
        
        for model_name, model_data in data["models"].items():
            models.append(model_name)
            response_times.append(model_data["avg_response_time"])
            quality_scores.append(model_data["avg_quality"])
        
        fig = go.Figure(data=go.Scatter(
            x=response_times,
            y=quality_scores,
            mode='markers+text',
            text=models,
            textposition="top center",
            marker=dict(
                size=12,
                color=quality_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Quality Score")
            )
        ))
        
        fig.update_layout(
            title="Performance Overview: Response Time vs Quality",
            xaxis_title="Response Time (seconds)",
            yaxis_title="Quality Score",
            height=500
        )
        
        return fig


def generate_enhanced_report(results: Dict[str, List[Any]], title: str = "LLM Test Report") -> str:
    """Convenience function to generate an enhanced report"""
    generator = EnhancedReportGenerator()
    return generator.generate_comprehensive_report(results, title)