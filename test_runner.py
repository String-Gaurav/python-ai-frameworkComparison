"""
Simple command-line interface for running LLM tests
"""

import argparse
import sys
import json
import datetime
import statistics
from pathlib import Path

# Import from our framework
try:
    from llm_test_framework import LLMTestFramework, TestCase
except ImportError:
    print("Error: Could not import llm_test_framework.py")
    print("Make sure llm_test_framework.py is in the same directory")
    sys.exit(1)


class Config:
    """Configuration settings for the LLM testing framework"""

    # Default models to test
    DEFAULT_OLLAMA_MODELS = [
        "llama3.2:latest",
        "mistral:7b"
    ]

    # Test categories to run by default
    DEFAULT_CATEGORIES = [
        "math",
        "programming",
        "creative",
        "knowledge",
        "reasoning"
    ]

    # Parallel execution settings
    MAX_WORKERS = 3
    ENABLE_PARALLEL = False

    # Output settings
    OUTPUT_DIR = "test_results"


def create_custom_test_suite(file_path: str):
    """Create a custom test suite from a JSON file"""
    sample_tests = [
        {
            "id": "custom_math_001",
            "category": "math",
            "prompt": "Calculate the compound interest on $1000 at 5% annually for 3 years",
            "expected_type": "math",
            "expected_answer": "$1157.63",
            "difficulty": "medium",
            "tags": ["finance", "compound_interest"]
        },
        {
            "id": "custom_code_001",
            "category": "programming",
            "prompt": "Write a Python function to implement binary search",
            "expected_type": "code",
            "difficulty": "medium",
            "tags": ["algorithms", "search"]
        },
        {
            "id": "custom_creative_001",
            "category": "creative",
            "prompt": "Write a product description for a smart water bottle",
            "expected_type": "creative",
            "difficulty": "easy",
            "tags": ["marketing", "product"]
        }
    ]

    with open(file_path, 'w') as f:
        json.dump(sample_tests, f, indent=2)

    print(f"Created sample test suite at {file_path}")


def generate_html_report(framework, results):
    """Generate an HTML report with visualizations"""
    try:
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .model-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e7f3ff; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Testing Report</h1>
        <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Session ID: {framework.session_id}</p>
    </div>
"""

        # Add performance metrics
        html_content += "<h2>Performance Overview</h2>\n"

        for model_name, model_results in results.items():
            if model_results:
                avg_time = statistics.mean([r.response_time for r in model_results])
                avg_quality = statistics.mean([r.scores.get('overall_quality', 0) for r in model_results])
                total_tests = len(model_results)

                html_content += f"""
    <div class="model-section">
        <h3>{model_name}</h3>
        <div class="metric">Avg Response Time: {avg_time:.2f}s</div>
        <div class="metric">Avg Quality Score: {avg_quality:.2f}</div>
        <div class="metric">Total Tests: {total_tests}</div>
    </div>
"""

        html_content += "</body></html>"

        # Save HTML report
        html_path = framework.output_dir / f"test_report_{framework.session_id}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to {html_path}")

    except Exception as e:
        print(f"Warning: Could not generate HTML report: {e}")


def main():
    parser = argparse.ArgumentParser(description="LLM Testing Framework CLI")

    # Model selection
    parser.add_argument("--models", nargs="+",
                        help="Ollama models to test (e.g., llama3.2:latest mistral:7b)")

    # Test selection
    parser.add_argument("--categories", nargs="+",
                        default=Config.DEFAULT_CATEGORIES,
                        help="Test categories to run")
    parser.add_argument("--custom-tests",
                        help="Path to custom test suite JSON file")
    parser.add_argument("--create-sample-tests",
                        help="Create sample custom test suite at specified path")

    # Execution options
    parser.add_argument("--parallel", action="store_true",
                        help="Run tests in parallel")
    parser.add_argument("--max-workers", type=int, default=Config.MAX_WORKERS,
                        help="Maximum number of parallel workers")

    # Output options
    parser.add_argument("--output-dir", default=Config.OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--generate-html", action="store_true",
                        help="Generate HTML report")

    args = parser.parse_args()

    # Create sample test suite if requested
    if args.create_sample_tests:
        create_custom_test_suite(args.create_sample_tests)
        return

    # Initialize framework
    framework = LLMTestFramework(output_dir=args.output_dir)

    # Add Ollama models
    if args.models:
        for model in args.models:
            try:
                framework.add_ollama_model(model)
                print(f"Added model: {model}")
            except Exception as e:
                print(f"Warning: Could not add Ollama model {model}: {e}")
    else:
        # Use default models
        for model in Config.DEFAULT_OLLAMA_MODELS:
            try:
                framework.add_ollama_model(model)
                print(f"Added default model: {model}")
            except Exception as e:
                print(f"Warning: Could not add default model {model}: {e}")

    # Load custom tests if provided
    if args.custom_tests:
        if Path(args.custom_tests).exists():
            framework.test_suite.load_from_file(args.custom_tests)
            print(f"Loaded custom tests from {args.custom_tests}")
        else:
            print(f"Error: Custom test file {args.custom_tests} not found")
            sys.exit(1)

    # Check if we have any models to test
    if not framework.providers:
        print("Error: No models available to test. Please check your model configurations.")
        print("\nTry:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check that your models are installed: ollama list")
        print("3. Download models if needed: ollama pull llama3.2:latest")
        sys.exit(1)

    print(f"\nStarting tests with {len(framework.providers)} models...")
    print(f"Testing categories: {', '.join(args.categories)}")
    print("=" * 60)

    # Run the test suite
    try:
        results = framework.run_test_suite(
            categories=args.categories,
            parallel=args.parallel,
            max_workers=args.max_workers
        )

        # Generate report
        report = framework.generate_report(results)
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(report)

        # Export options
        if args.export_csv:
            framework.export_results_csv()
            print("✅ Results exported to CSV")

        if args.generate_html:
            generate_html_report(framework, results)
            print("✅ HTML report generated")

        print(f"\n🎉 All results saved to: {framework.output_dir}")
        print("\nNext steps:")
        print("- View detailed results in the generated files")
        print("- Run with --export-csv for spreadsheet analysis")
        print("- Run with --generate-html for visual reports")

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during test execution: {e}")
        print("\nTroubleshooting:")
        print("1. Check that Ollama is running: ollama serve")
        print("2. Verify models are available: ollama list")
        print("3. Test individual model: ollama run llama3.2:latest 'hello'")
        sys.exit(1)


if __name__ == "__main__":
    main()