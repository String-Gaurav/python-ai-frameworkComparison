"""
Minimal LLM Testing Framework
Save this as: llm_test_framework.py
"""

import json
import time
import statistics
import datetime
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case"""
    id: str
    category: str
    prompt: str
    expected_type: str
    expected_answer: Optional[str] = None
    difficulty: str = "medium"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TestResult:
    """Represents the result of a single test execution"""
    test_id: str
    model_name: str
    response: str
    response_time: float
    timestamp: str
    scores: Dict[str, float]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> Tuple[str, float]:
        """Generate response and return (response, response_time)"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name"""
        pass


class OllamaProvider(ModelProvider):
    """Ollama model provider"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            import ollama
            self.client = ollama
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")

    def generate_response(self, prompt: str, **kwargs) -> Tuple[str, float]:
        start_time = time.time()
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                **kwargs
            )
            end_time = time.time()
            return response['message']['content'], end_time - start_time
        except Exception as e:
            logger.error(f"Error with Ollama model {self.model_name}: {e}")
            return f"Error: {e}", 0.0

    def get_model_name(self) -> str:
        return self.model_name


class ResponseScorer:
    """Automated response scoring system"""

    @staticmethod
    def score_response_length(response: str) -> float:
        """Score based on response length appropriateness"""
        length = len(response.split())
        if 10 <= length <= 200:
            return 1.0
        elif length < 10:
            return max(0.0, length / 10)
        else:
            return max(0.3, 1.0 - (length - 200) / 1000)

    @staticmethod
    def score_code_quality(response: str) -> float:
        """Score code responses for basic quality indicators"""
        code_indicators = ['def ', 'class ', 'import ', 'return', 'if ', 'for ', 'while ']
        syntax_indicators = ['(', ')', '{', '}', '[', ']', ':', ';']

        code_score = sum(1 for indicator in code_indicators if indicator in response)
        syntax_score = sum(1 for indicator in syntax_indicators if indicator in response)

        total_score = (code_score * 0.6 + syntax_score * 0.4) / 10
        return min(1.0, total_score)

    @staticmethod
    def score_factual_confidence(response: str) -> float:
        """Score factual responses for confidence indicators"""
        confident_phrases = ['is', 'are', 'was', 'were', 'will', 'can', 'does']
        uncertain_phrases = ['might', 'maybe', 'possibly', 'perhaps', 'could be', 'may']

        confident_count = sum(1 for phrase in confident_phrases if phrase in response.lower())
        uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in response.lower())

        if confident_count + uncertain_count == 0:
            return 0.5

        confidence_ratio = confident_count / (confident_count + uncertain_count)
        return confidence_ratio

    @staticmethod
    def score_creativity(response: str) -> float:
        """Score creative responses for uniqueness and engagement"""
        creative_indicators = [
            'imagine', 'story', 'once upon', 'suddenly', 'beautiful', 'magical',
            'adventure', 'journey', 'dream', 'wonder', 'amazing', 'incredible'
        ]

        creative_count = sum(1 for indicator in creative_indicators if indicator in response.lower())
        creativity_score = min(1.0, creative_count / 5)

        # Bonus for varied sentence length
        sentences = response.split('.')
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                variance = statistics.variance(lengths) if len(lengths) > 1 else 0
                variety_bonus = min(0.3, variance / 100)
                creativity_score += variety_bonus

        return min(1.0, creativity_score)

    @classmethod
    def score_response(cls, response: str, test_case: TestCase) -> Dict[str, float]:
        """Comprehensive response scoring"""
        scores = {
            'length_score': cls.score_response_length(response),
            'overall_quality': 0.5  # Default baseline
        }

        # Category-specific scoring
        if test_case.expected_type == 'code':
            scores['code_quality'] = cls.score_code_quality(response)
            scores['overall_quality'] = (scores['length_score'] + scores['code_quality']) / 2

        elif test_case.expected_type == 'factual':
            scores['confidence'] = cls.score_factual_confidence(response)
            scores['overall_quality'] = (scores['length_score'] + scores['confidence']) / 2

        elif test_case.expected_type == 'creative':
            scores['creativity'] = cls.score_creativity(response)
            scores['overall_quality'] = (scores['length_score'] + scores['creativity']) / 2

        elif test_case.expected_type == 'math':
            # Simple math validation (look for numbers and operators)
            math_indicators = any(char in response for char in '0123456789+-*/=')
            scores['math_content'] = 1.0 if math_indicators else 0.0
            scores['overall_quality'] = (scores['length_score'] + scores['math_content']) / 2

        return scores


class DatabaseManager:
    """Manages test results database"""

    def __init__(self, db_path: str = "llm_test_results.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                model_name TEXT,
                category TEXT,
                prompt TEXT,
                response TEXT,
                response_time REAL,
                timestamp TEXT,
                scores TEXT,
                metadata TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def save_result(self, result: TestResult, test_case: TestCase):
        """Save a test result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO test_results 
            (test_id, model_name, category, prompt, response, response_time, timestamp, scores, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.test_id,
            result.model_name,
            test_case.category,
            test_case.prompt,
            result.response,
            result.response_time,
            result.timestamp,
            json.dumps(result.scores),
            json.dumps(result.metadata)
        ))

        conn.commit()
        conn.close()

    def get_results(self, model_name: str = None, category: str = None) -> List[Dict]:
        """Retrieve test results with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if category:
            query += " AND category = ?"
            params.append(category)

        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results


class TestSuiteManager:
    """Manages test cases and test suites"""

    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.load_default_tests()

    def load_default_tests(self):
        """Load default test cases"""
        default_tests = [
            # Math and Logic
            TestCase("math_001", "math", "What is 15% of 240?", "math", "36"),
            TestCase("math_002", "math", "If I have 100 apples and give away 25%, then eat 10 more, how many are left?",
                     "math", "65"),
            TestCase("logic_001", "reasoning", "All cats are animals. Fluffy is a cat. What can we conclude?",
                     "factual", "Fluffy is an animal"),

            # Code Generation
            TestCase("code_001", "programming", "Write a Python function to reverse a string", "code"),
            TestCase("code_002", "programming", "Create a function that checks if a number is prime", "code"),
            TestCase("code_003", "programming", "Write a simple calculator function in Python", "code"),

            # Creative Writing
            TestCase("creative_001", "creative", "Write a short story about a time-traveling scientist", "creative"),
            TestCase("creative_002", "creative", "Compose a haiku about artificial intelligence", "creative"),
            TestCase("creative_003", "creative", "Describe a futuristic city in 3 sentences", "creative"),

            # Factual Knowledge
            TestCase("fact_001", "knowledge", "What is the capital of Australia?", "factual", "Canberra"),
            TestCase("fact_002", "knowledge", "Explain photosynthesis in simple terms", "factual"),
            TestCase("fact_003", "knowledge", "What are the three states of matter?", "factual", "Solid, liquid, gas"),

            # Reasoning and Problem Solving
            TestCase("reason_001", "reasoning",
                     "A farmer has chickens and cows. There are 30 heads and 74 legs total. How many chickens?", "math",
                     "23"),
            TestCase("reason_002", "reasoning", "What comes next in this sequence: 2, 4, 8, 16, ?", "math", "32"),
        ]

        self.test_cases.extend(default_tests)

    def add_test_case(self, test_case: TestCase):
        """Add a new test case"""
        self.test_cases.append(test_case)

    def get_tests_by_category(self, category: str) -> List[TestCase]:
        """Get test cases by category"""
        return [tc for tc in self.test_cases if tc.category == category]


class LLMTestFramework:
    """Main testing framework class"""

    def __init__(self, output_dir: str = "test_results"):
        self.providers: List[ModelProvider] = []
        self.test_suite = TestSuiteManager()
        self.scorer = ResponseScorer()
        self.db = DatabaseManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Session tracking
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[TestResult] = []

    def add_provider(self, provider: ModelProvider):
        """Add a model provider to test"""
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.get_model_name()}")

    def add_ollama_model(self, model_name: str):
        """Convenience method to add Ollama model"""
        self.add_provider(OllamaProvider(model_name))

    def run_single_test(self, provider: ModelProvider, test_case: TestCase) -> TestResult:
        """Run a single test case on a provider"""
        logger.info(f"Running test {test_case.id} on {provider.get_model_name()}")

        response, response_time = provider.generate_response(test_case.prompt)
        scores = self.scorer.score_response(response, test_case)

        result = TestResult(
            test_id=test_case.id,
            model_name=provider.get_model_name(),
            response=response,
            response_time=response_time,
            timestamp=datetime.datetime.now().isoformat(),
            scores=scores,
            metadata={
                'category': test_case.category,
                'difficulty': test_case.difficulty,
                'expected_type': test_case.expected_type
            }
        )

        # Save to database
        self.db.save_result(result, test_case)
        self.results.append(result)

        return result

    def run_test_suite(self,
                       categories: List[str] = None,
                       parallel: bool = False,
                       max_workers: int = 3) -> Dict[str, List[TestResult]]:
        """Run the complete test suite"""
        logger.info(f"Starting test suite with {len(self.providers)} providers")

        # Filter test cases
        test_cases = self.test_suite.test_cases
        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]

        results_by_model = {}

        for provider in self.providers:
            model_name = provider.get_model_name()
            results_by_model[model_name] = []

            for test_case in test_cases:
                result = self.run_single_test(provider, test_case)
                results_by_model[model_name].append(result)

        logger.info("Test suite completed")
        return results_by_model

    def generate_report(self, results: Dict[str, List[TestResult]] = None) -> str:
        """Generate a comprehensive test report"""
        if results is None:
            # Get all results from database
            all_results = self.db.get_results()
            results = {}
            for result in all_results:
                model = result['model_name']
                if model not in results:
                    results[model] = []

                test_result = TestResult(
                    test_id=result['test_id'],
                    model_name=result['model_name'],
                    response=result['response'],
                    response_time=result['response_time'],
                    timestamp=result['timestamp'],
                    scores=json.loads(result['scores']),
                    metadata=json.loads(result['metadata'])
                )
                results[model].append(test_result)

        report = f"""
# LLM Testing Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}

## Executive Summary
"""

        # Summary statistics
        total_tests = sum(len(model_results) for model_results in results.values())
        models_tested = list(results.keys())

        report += f"- **Total Tests Run**: {total_tests}\n"
        report += f"- **Models Tested**: {len(models_tested)}\n"
        report += f"- **Models**: {', '.join(models_tested)}\n\n"

        # Performance comparison
        report += "## Performance Comparison\n\n"
        report += "| Model | Avg Response Time | Avg Quality Score | Total Tests |\n"
        report += "|-------|------------------|-------------------|-------------|\n"

        for model_name, model_results in results.items():
            if model_results:
                avg_time = statistics.mean([r.response_time for r in model_results])
                avg_quality = statistics.mean([r.scores.get('overall_quality', 0) for r in model_results])
                test_count = len(model_results)

                report += f"| {model_name} | {avg_time:.2f}s | {avg_quality:.2f} | {test_count} |\n"

        # Save report
        report_path = self.output_dir / f"test_report_{self.session_id}.md"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {report_path}")
        return report

    def export_results_csv(self, filename: str = None):
        """Export results to CSV"""
        if filename is None:
            filename = self.output_dir / f"test_results_{self.session_id}.csv"

        all_results = self.db.get_results()

        import csv
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['test_id', 'model_name', 'category', 'response_time', 'overall_quality', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in all_results:
                scores = json.loads(result['scores'])
                writer.writerow({
                    'test_id': result['test_id'],
                    'model_name': result['model_name'],
                    'category': result['category'],
                    'response_time': result['response_time'],
                    'overall_quality': scores.get('overall_quality', 0),
                    'timestamp': result['timestamp']
                })

        logger.info(f"Results exported to {filename}")


# Example usage
def main():
    """Example usage"""
    framework = LLMTestFramework()

    # Add models
    framework.add_ollama_model("llama3.2:latest")

    # Run tests
    results = framework.run_test_suite(categories=["math"])

    # Generate report
    report = framework.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()