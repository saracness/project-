"""
ReportGenerator - Automatic PDF/HTML report generation
"""
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class ReportGenerator:
    """
    Generate automatic experiment reports.

    Features:
    - PDF reports (using text-based layout)
    - HTML reports
    - Auto-generation at intervals
    - Custom templates
    """

    def __init__(self,
                 db_path: str,
                 output_dir: str = 'outputs/reports',
                 auto_generate: bool = False,
                 interval: int = 500):
        """
        Initialize report generator.

        Args:
            db_path: Database path
            output_dir: Output directory
            auto_generate: Enable auto-generation
            interval: Generation interval (episodes)
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_generate = auto_generate
        self.interval = interval

        self.last_generation = 0

    def generate_text_report(self, experiment_name: str = "experiment") -> str:
        """
        Generate text-based report.

        Args:
            experiment_name: Name of experiment

        Returns:
            Report file path
        """
        from .statistical_analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer(str(self.db_path))

        # Generate report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("=" * 80)
        lines.append(f"EXPERIMENT REPORT: {experiment_name}")
        lines.append(f"Generated: {timestamp}")
        lines.append("=" * 80)
        lines.append("")

        # Statistical analysis
        lines.append(analyzer.generate_report())

        # Save to file
        output_path = self.output_dir / f"{experiment_name}_{int(time.time())}.txt"

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✅ Generated report: {output_path}")
        return str(output_path)

    def generate_html_report(self, experiment_name: str = "experiment") -> str:
        """
        Generate HTML report with embedded graphs.

        Args:
            experiment_name: Name of experiment

        Returns:
            Report file path
        """
        from .statistical_analyzer import StatisticalAnalyzer
        from .scientific_plotter import ScientificPlotter

        # Generate graphs first
        plotter = ScientificPlotter(str(self.db_path))
        graph_dir = self.output_dir / 'graphs_temp'
        graph_dir.mkdir(exist_ok=True)

        plotter.plot_population_dynamics(
            save_path=str(graph_dir / 'population.png')
        )

        # Generate statistics
        analyzer = StatisticalAnalyzer(str(self.db_path))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Report: {experiment_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .graph {{
            text-align: center;
            margin: 20px 0;
        }}
        .graph img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .stats-table th, .stats-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .stats-table th {{
            background-color: #667eea;
            color: white;
        }}
        .stats-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Experiment Report: {experiment_name}</h1>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="section">
        <h2>📊 Population Dynamics</h2>
        <div class="graph">
            <img src="graphs_temp/population.png" alt="Population Dynamics">
        </div>
    </div>

    <div class="section">
        <h2>📈 Statistical Summary</h2>
        <pre style="background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto;">
{analyzer.generate_report()}
        </pre>
    </div>

    <div class="footer">
        <p>MicroLife Scientific Analysis System</p>
        <p>Auto-generated report - {timestamp}</p>
    </div>
</body>
</html>
"""

        # Save HTML
        output_path = self.output_dir / f"{experiment_name}_{int(time.time())}.html"

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✅ Generated HTML report: {output_path}")
        return str(output_path)

    def check_auto_generate(self, current_episode: int):
        """
        Check if report should be auto-generated.

        Args:
            current_episode: Current episode number
        """
        if not self.auto_generate:
            return

        if current_episode > 0 and current_episode % self.interval == 0:
            if current_episode != self.last_generation:
                self.last_generation = current_episode
                self.generate_html_report(f"auto_ep{current_episode}")
