#!/usr/bin/env python3
"""
Simple script to export your existing test results
Save this as: export_my_results.py
"""

import sqlite3
import csv
import json
import datetime
from pathlib import Path


def find_database():
    """Find the test results database"""
    possible_paths = [
        "llm_test_results.db",
        "../llm_test_results.db",
        "/Users/saurabh/PyCharmMiscProject/llm_test_results.db",
        "test_results/llm_test_results.db"
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found database: {path}")
            return path

    print("❌ Could not find database file")
    print("Searching for .db files...")
    for file in Path("..").rglob("*.db"):
        print(f"  Found: {file}")
    return None


def export_to_csv(db_path):
    """Export your test results to CSV"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all results
    cursor.execute("SELECT * FROM test_results ORDER BY timestamp DESC")
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    if not results:
        print("❌ No test results found in database")
        return

    output_file = f"llm_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Custom header for better readability
        writer.writerow([
            'Test ID', 'Model', 'Category', 'Prompt', 'Response Time (s)',
            'Quality Score', 'Test Date', 'Response Preview'
        ])

        # Process each result
        for row in results:
            result_dict = dict(zip(columns, row))
            scores = json.loads(result_dict['scores'])

            # Clean and truncate text fields
            prompt = result_dict['prompt'].replace('\n', ' ')[:100]
            if len(result_dict['prompt']) > 100:
                prompt += "..."

            response = result_dict['response'].replace('\n', ' ')[:200]
            if len(result_dict['response']) > 200:
                response += "..."

            writer.writerow([
                result_dict['test_id'],
                result_dict['model_name'],
                result_dict['category'],
                prompt,
                round(result_dict['response_time'], 2),
                round(scores.get('overall_quality', 0), 3),
                result_dict['timestamp'][:19],  # Remove microseconds
                response
            ])

    conn.close()
    print(f"✅ Exported {len(results)} results to {output_file}")

    # Show summary
    print(f"\n📊 Export Summary:")
    print(f"   Total tests: {len(results)}")

    # Count by model
    cursor = sqlite3.connect(db_path).cursor()
    cursor.execute("SELECT model_name, COUNT(*) FROM test_results GROUP BY model_name")
    for model, count in cursor.fetchall():
        print(f"   {model}: {count} tests")

    return output_file


def create_summary_report():
    """Create a quick summary based on your markdown reports"""

    # Your actual results from the reports
    results_data = {
        "Session 1 (Basic Test)": {
            "date": "2025-07-12 01:38:55",
            "models": ["llama3.2:latest"],
            "total_tests": 2,
            "results": {
                "llama3.2:latest": {"avg_time": 2.95, "quality": 1.00, "tests": 2}
            }
        },
        "Session 2 (Full Comparison)": {
            "date": "2025-07-12 01:42:22",
            "models": ["llama3.2:latest", "mistral:7b"],
            "total_tests": 28,
            "results": {
                "llama3.2:latest": {"avg_time": 4.04, "quality": 0.83, "tests": 14},
                "mistral:7b": {"avg_time": 7.10, "quality": 0.83, "tests": 14}
            }
        }
    }

    # Create HTML summary
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Testing Results Summary</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .header h1 {{ color: #333; margin: 0; font-size: 2.5em; }}
        .header p {{ color: #666; margin: 10px 0; }}
        .session {{ margin: 30px 0; padding: 25px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #007bff; }}
        .session h3 {{ color: #007bff; margin-top: 0; }}
        .comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .model-card {{ flex: 1; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .model-card.winner {{ border: 2px solid #28a745; background: #f8fff9; }}
        .model-card h4 {{ color: #333; margin: 0 0 15px 0; }}
        .metric {{ display: flex; justify-content: space-between; margin: 8px 0; }}
        .metric strong {{ color: #007bff; }}
        .winner-badge {{ background: #28a745; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }}
        .insights {{ margin: 30px 0; padding: 20px; background: #e8f4f8; border-radius: 8px; }}
        .insights h3 {{ color: #0c5460; margin-top: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LLM Testing Results</h1>
            <p>Comprehensive Model Performance Analysis</p>
            <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""

    for session_name, session_data in results_data.items():
        html_content += f"""
        <div class="session">
            <h3>{session_name}</h3>
            <p><strong>Date:</strong> {session_data['date']} | <strong>Total Tests:</strong> {session_data['total_tests']}</p>

            <div class="comparison">
"""

        # Find the winner (fastest model)
        fastest_model = min(session_data['results'].keys(),
                            key=lambda x: session_data['results'][x]['avg_time'])

        for model, stats in session_data['results'].items():
            is_winner = model == fastest_model and len(session_data['results']) > 1
            winner_class = " winner" if is_winner else ""

            html_content += f"""
                <div class="model-card{winner_class}">
                    <h4>{model} {' <span class="winner-badge">⚡ Fastest</span>' if is_winner else ''}</h4>
                    <div class="metric">
                        <span>Response Time:</span>
                        <strong>{stats['avg_time']}s</strong>
                    </div>
                    <div class="metric">
                        <span>Quality Score:</span>
                        <strong>{stats['quality']}</strong>
                    </div>
                    <div class="metric">
                        <span>Tests Run:</span>
                        <strong>{stats['tests']}</strong>
                    </div>
                </div>
"""

        html_content += """
            </div>
        </div>
"""

    html_content += f"""
        <div class="insights">
            <h3>🎯 Key Insights</h3>
            <ul>
                <li><strong>Speed Champion:</strong> llama3.2:latest is 76% faster than mistral:7b (4.04s vs 7.10s)</li>
                <li><strong>Quality Tie:</strong> Both models achieved identical quality scores (0.83)</li>
                <li><strong>Efficiency Winner:</strong> llama3.2 delivers same quality with half the model size</li>
                <li><strong>Consistency:</strong> llama3.2 maintained good performance across {results_data['Session 2 (Full Comparison)']['total_tests']} tests</li>
                <li><strong>Recommendation:</strong> Use llama3.2:latest for production - better speed, same quality, lower resource usage</li>
            </ul>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>Generated by LLM Testing Framework | Professional AI Model Evaluation</p>
        </div>
    </div>
</body>
</html>
"""

    output_file = f"llm_test_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Created HTML summary: {output_file}")
    return output_file


def main():
    print("📊 LLM Test Results Exporter")
    print("=" * 40)

    # Try to export from database
    db_path = find_database()

    if db_path:
        print("\n📁 Exporting database results...")
        csv_file = export_to_csv(db_path)
    else:
        print("\n⚠️  Database not found, will create summary from reports only")

    # Create summary report regardless
    print("\n📝 Creating summary report...")
    html_file = create_summary_report()

    print(f"\n🎉 Export Complete!")
    print(f"Files created:")
    if db_path:
        print(f"  📄 CSV Export: {csv_file}")
    print(f"  🌐 HTML Summary: {html_file}")

    print(f"\n💡 Next steps:")
    print(f"  - Open the HTML file in your browser for visual analysis")
    if db_path:
        print(f"  - Import the CSV into Excel/Google Sheets for detailed analysis")
    print(f"  - Use these results to choose the best model for your needs")


if __name__ == "__main__":
    main()