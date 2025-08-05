#!/usr/bin/env python3
"""
Enhanced CLI Interface for LLM Testing Framework
Provides comprehensive command-line options for running tests and managing results
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from llm_test_framework import LLMTestFramework, TestCase
from config import config
from model_providers import list_available_providers, get_provider
from parallel_runner import ParallelTestRunner, ParallelTestConfig
from enhanced_reporting import generate_enhanced_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Testing Framework - Compare and evaluate language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --models llama3.2 mistral:7b --categories math code
  %(prog)s run --config custom_config.json --parallel --workers 4
  %(prog)s list-models --provider ollama
  %(prog)s report --format html --output results.html
  %(prog)s benchmark --models gpt-3.5-turbo claude-3-haiku --iterations 5
        """
    )
    
    # Global options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: from config)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run LLM tests')
    run_parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='Models to test (e.g., llama3.2 mistral:7b gpt-3.5-turbo)'
    )
    run_parser.add_argument(
        '--categories',
        nargs='+',
        default=['math', 'programming', 'creative', 'knowledge', 'reasoning'],
        help='Test categories to run'
    )
    run_parser.add_argument(
        '--test-file',
        type=str,
        help='Custom test file (JSON format)'
    )
    run_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel execution'
    )
    run_parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers (default: 3)'
    )
    run_parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout per test in seconds (default: 120)'
    )
    run_parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Number of retries for failed tests (default: 3)'
    )
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.add_argument(
        '--provider',
        choices=['ollama', 'openai', 'anthropic', 'all'],
        default='all',
        help='Provider to list models for'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports from existing results')
    report_parser.add_argument(
        '--format',
        choices=['html', 'csv', 'json', 'markdown', 'enhanced'],
        default='enhanced',
        help='Report format'
    )
    report_parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    report_parser.add_argument(
        '--filter-model',
        type=str,
        help='Filter results by model name'
    )
    report_parser.add_argument(
        '--filter-category',
        type=str,
        help='Filter results by category'
    )
    report_parser.add_argument(
        '--since',
        type=str,
        help='Filter results since date (YYYY-MM-DD)'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='Models to benchmark'
    )
    benchmark_parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of benchmark iterations (default: 3)'
    )
    benchmark_parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='Number of warmup iterations (default: 1)'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    config_subparsers.add_parser('show', help='Show current configuration')
    
    set_parser = config_subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('key', help='Configuration key (dot notation)')
    set_parser.add_argument('value', help='Configuration value')
    
    get_parser = config_subparsers.add_parser('get', help='Get configuration value')
    get_parser.add_argument('key', help='Configuration key (dot notation)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument(
        '--check-models',
        action='store_true',
        help='Check model availability'
    )
    
    return parser


def setup_logging(verbose: bool):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Suppress verbose logs from third-party libraries unless in debug mode
    if not verbose:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('anthropic').setLevel(logging.WARNING)


def load_custom_tests(test_file: str) -> List[TestCase]:
    """Load custom test cases from JSON file"""
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        test_cases = []
        for test in test_data:
            test_case = TestCase(
                id=test['id'],
                category=test.get('category', 'custom'),
                prompt=test['prompt'],
                expected_type=test.get('expected_type', 'text'),
                expected_answer=test.get('expected_answer'),
                difficulty=test.get('difficulty', 'medium'),
                tags=test.get('tags', [])
            )
            test_cases.append(test_case)
        
        logger.info(f"Loaded {len(test_cases)} custom test cases from {test_file}")
        return test_cases
        
    except Exception as e:
        logger.error(f"Failed to load custom tests from {test_file}: {e}")
        sys.exit(1)


def run_tests(args) -> int:
    """Run LLM tests based on arguments"""
    try:
        # Initialize framework
        framework = LLMTestFramework(output_dir=args.output_dir)
        
        # Add models
        for model_name in args.models:
            if '/' in model_name or model_name.startswith('gpt') or model_name.startswith('claude'):
                # Attempt to determine provider
                if model_name.startswith('gpt'):
                    provider = get_provider('openai')
                    if provider.is_available():
                        framework.add_model(model_name, provider)
                    else:
                        logger.error(f"OpenAI provider not available for model {model_name}")
                        continue
                elif model_name.startswith('claude'):
                    provider = get_provider('anthropic')
                    if provider.is_available():
                        framework.add_model(model_name, provider)
                    else:
                        logger.error(f"Anthropic provider not available for model {model_name}")
                        continue
                else:
                    # Assume Ollama for other models
                    framework.add_ollama_model(model_name)
            else:
                # Default to Ollama
                framework.add_ollama_model(model_name)
        
        # Load test cases
        if args.test_file:
            test_cases = load_custom_tests(args.test_file)
            results = framework.run_custom_tests(test_cases)
        else:
            results = framework.run_test_suite(categories=args.categories)
        
        # Generate reports
        framework.generate_report(results)
        
        if any(results.values()):
            generate_enhanced_report(results, "LLM Test Results")
        
        logger.info("Tests completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


def list_models(args) -> int:
    """List available models"""
    try:
        providers = list_available_providers()
        
        if args.provider == 'all':
            for provider_name, available in providers.items():
                print(f"\n{provider_name.upper()} Provider: {'✓ Available' if available else '✗ Not Available'}")
                if available:
                    try:
                        provider = get_provider(provider_name)
                        models = provider.list_models()
                        for model in models:
                            print(f"  - {model}")
                    except Exception as e:
                        print(f"  Error listing models: {e}")
        else:
            provider_name = args.provider
            if provider_name in providers and providers[provider_name]:
                provider = get_provider(provider_name)
                models = provider.list_models()
                print(f"{provider_name.upper()} Models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print(f"{provider_name.upper()} provider not available")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return 1


def generate_report_cmd(args) -> int:
    """Generate reports from existing results"""
    try:
        framework = LLMTestFramework(output_dir=args.output_dir)
        
        # Get results from database with filters
        filters = {}
        if args.filter_model:
            filters['model_name'] = args.filter_model
        if args.filter_category:
            filters['category'] = args.filter_category
        if args.since:
            filters['since'] = args.since
        
        # TODO: Implement database filtering
        db_results = framework.db.get_results()
        
        if not db_results:
            logger.warning("No results found in database")
            return 1
        
        # Convert to framework format
        results = {}
        for result in db_results:
            model = result['model_name']
            if model not in results:
                results[model] = []
            # Convert back to TestResult objects
            # TODO: Implement proper conversion
        
        if args.format == 'enhanced':
            report_path = generate_enhanced_report(results, "Generated Report")
            print(f"Enhanced report generated: {report_path}")
        else:
            report = framework.generate_report(results)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"Report saved to: {args.output}")
            else:
                print(report)
        
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1


def benchmark_models(args) -> int:
    """Run performance benchmarks"""
    try:
        logger.info(f"Running benchmarks for models: {args.models}")
        
        # Create simple benchmark test
        benchmark_test = TestCase(
            id="benchmark_001",
            category="benchmark",
            prompt="Write a Python function to calculate the factorial of a number.",
            expected_type="code"
        )
        
        framework = LLMTestFramework(output_dir=args.output_dir)
        
        # Add models
        for model_name in args.models:
            framework.add_ollama_model(model_name)  # Simplified for now
        
        # Run multiple iterations
        all_results = []
        for iteration in range(args.iterations):
            logger.info(f"Running benchmark iteration {iteration + 1}/{args.iterations}")
            results = framework.run_custom_tests([benchmark_test])
            all_results.append(results)
        
        # Analyze performance
        print("\nBenchmark Results:")
        print("=" * 50)
        
        for model_name in args.models:
            times = []
            for result_set in all_results:
                if model_name in result_set and result_set[model_name]:
                    times.append(result_set[model_name][0].response_time)
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"{model_name}:")
                print(f"  Average: {avg_time:.2f}s")
                print(f"  Min: {min_time:.2f}s")
                print(f"  Max: {max_time:.2f}s")
                print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


def manage_config(args) -> int:
    """Manage configuration"""
    try:
        if args.config_action == 'show':
            print(json.dumps(config.config, indent=2))
        elif args.config_action == 'get':
            value = config.get(args.key)
            print(f"{args.key}: {value}")
        elif args.config_action == 'set':
            config.set(args.key, args.value)
            config.save_config()
            print(f"Set {args.key} = {args.value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Config management failed: {e}")
        return 1


def show_status(args) -> int:
    """Show system status"""
    try:
        print("LLM Testing Framework Status")
        print("=" * 40)
        
        # Check providers
        providers = list_available_providers()
        print("\nProviders:")
        for name, available in providers.items():
            status = "✓" if available else "✗"
            print(f"  {status} {name.capitalize()}")
        
        # Check models if requested
        if args.check_models:
            print("\nModel Availability:")
            for provider_name, available in providers.items():
                if available:
                    try:
                        provider = get_provider(provider_name)
                        models = provider.list_models()
                        print(f"  {provider_name.capitalize()}: {len(models)} models available")
                    except Exception as e:
                        print(f"  {provider_name.capitalize()}: Error - {e}")
        
        # Check output directory
        output_dir = Path(config.get("output.directory", "test_results"))
        print(f"\nOutput directory: {output_dir}")
        print(f"  Exists: {'✓' if output_dir.exists() else '✗'}")
        if output_dir.exists():
            db_files = list(output_dir.glob("*.db"))
            csv_files = list(output_dir.glob("*.csv"))
            html_files = list(output_dir.glob("*.html"))
            print(f"  Database files: {len(db_files)}")
            print(f"  CSV files: {len(csv_files)}")
            print(f"  HTML files: {len(html_files)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    if Path(args.config).exists():
        config.__init__(args.config)
    
    # Set output directory if provided
    if args.output_dir:
        config.set("output.directory", args.output_dir)
    
    # Execute command
    try:
        if args.command == 'run':
            return run_tests(args)
        elif args.command == 'list-models':
            return list_models(args)
        elif args.command == 'report':
            return generate_report_cmd(args)
        elif args.command == 'benchmark':
            return benchmark_models(args)
        elif args.command == 'config':
            return manage_config(args)
        elif args.command == 'status':
            return show_status(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())