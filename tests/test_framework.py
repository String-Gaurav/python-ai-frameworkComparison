"""
Unit tests for the LLM Testing Framework
"""

import unittest
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('..')

from llm_test_framework import (
    LLMTestFramework, TestCase, TestResult, 
    ResponseScorer, DatabaseManager
)
from config import Config


class TestLLMTestFramework(unittest.TestCase):
    """Test cases for the main LLM Test Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = LLMTestFramework(output_dir=self.temp_dir)
        
        # Create test case
        self.test_case = TestCase(
            id="test_math_001",
            category="math",
            prompt="What is 2 + 2?",
            expected_type="numeric",
            expected_answer="4",
            difficulty="easy",
            tags=["arithmetic", "basic"]
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_test_case_creation(self):
        """Test TestCase dataclass creation"""
        self.assertEqual(self.test_case.id, "test_math_001")
        self.assertEqual(self.test_case.category, "math")
        self.assertEqual(self.test_case.prompt, "What is 2 + 2?")
        self.assertEqual(self.test_case.expected_answer, "4")
        self.assertEqual(self.test_case.difficulty, "easy")
        self.assertIn("arithmetic", self.test_case.tags)
    
    def test_test_result_creation(self):
        """Test TestResult dataclass creation"""
        result = TestResult(
            test_id="test_001",
            model_name="test_model",
            response="The answer is 4",
            response_time=1.5,
            timestamp="2024-01-01T00:00:00",
            scores={"overall_quality": 0.85}
        )
        
        self.assertEqual(result.test_id, "test_001")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.response, "The answer is 4")
        self.assertEqual(result.response_time, 1.5)
        self.assertEqual(result.scores["overall_quality"], 0.85)
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        self.assertIsInstance(self.framework.db, DatabaseManager)
        self.assertIsInstance(self.framework.scorer, ResponseScorer)
        self.assertTrue(Path(self.temp_dir).exists())
    
    @patch('ollama.chat')
    def test_add_ollama_model(self, mock_chat):
        """Test adding Ollama model"""
        mock_chat.return_value = {
            'message': {'content': 'Test response'}
        }
        
        self.framework.add_ollama_model("test_model")
        self.assertIn("test_model", self.framework.models)
        
        # Test model can generate response
        provider = self.framework.models["test_model"]
        result = provider.generate_response("test prompt", "test_model")
        self.assertEqual(result["response"], "Test response")


class TestResponseScorer(unittest.TestCase):
    """Test cases for the ResponseScorer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = ResponseScorer()
    
    def test_length_score(self):
        """Test length scoring"""
        # Good length response
        good_response = "This is a reasonably sized response that should score well."
        score = self.scorer._calculate_length_score(good_response)
        self.assertGreater(score, 0.5)
        
        # Too short response
        short_response = "No."
        score = self.scorer._calculate_length_score(short_response)
        self.assertLess(score, 0.5)
        
        # Too long response
        long_response = "a" * 2000
        score = self.scorer._calculate_length_score(long_response)
        self.assertLess(score, 0.5)
    
    def test_code_quality_score(self):
        """Test code quality scoring"""
        # Good code response
        code_response = """
        def add(a, b):
            return a + b
        
        result = add(2, 2)
        print(result)
        """
        score = self.scorer._calculate_code_quality(code_response)
        self.assertGreater(score, 0.7)
        
        # Non-code response
        text_response = "This is just plain text with no code."
        score = self.scorer._calculate_code_quality(text_response)
        self.assertLess(score, 0.3)
    
    def test_confidence_score(self):
        """Test confidence scoring"""
        # Confident response
        confident_response = "The answer is definitely 42. This is correct."
        score = self.scorer._calculate_confidence(confident_response)
        self.assertGreater(score, 0.6)
        
        # Uncertain response
        uncertain_response = "I think maybe it could be 42, but I'm not sure."
        score = self.scorer._calculate_confidence(uncertain_response)
        self.assertLess(score, 0.5)
    
    def test_creativity_score(self):
        """Test creativity scoring"""
        # Creative response
        creative_response = """
        Once upon a time, in a magical kingdom far away, there lived a brave knight
        who embarked on an extraordinary adventure to find the legendary crystal.
        """
        score = self.scorer._calculate_creativity(creative_response)
        self.assertGreater(score, 0.6)
        
        # Plain response
        plain_response = "The answer is 42."
        score = self.scorer._calculate_creativity(plain_response)
        self.assertLess(score, 0.4)
    
    def test_math_content_score(self):
        """Test math content scoring"""
        # Mathematical response
        math_response = "2 + 2 = 4, and sqrt(16) = 4, therefore the answer is 4."
        score = self.scorer._calculate_math_content(math_response)
        self.assertGreater(score, 0.7)
        
        # Non-mathematical response
        text_response = "This response has no mathematical content."
        score = self.scorer._calculate_math_content(text_response)
        self.assertLess(score, 0.3)
    
    def test_score_response_integration(self):
        """Test complete response scoring"""
        test_case = TestCase(
            id="test_001",
            category="math",
            prompt="What is 2 + 2?",
            expected_type="numeric"
        )
        
        response = "The answer to 2 + 2 is 4. This is basic arithmetic."
        scores = self.scorer.score_response(response, test_case)
        
        self.assertIn("overall_quality", scores)
        self.assertIn("length_score", scores)
        self.assertIn("math_content", scores)
        self.assertGreaterEqual(scores["overall_quality"], 0)
        self.assertLessEqual(scores["overall_quality"], 1)


class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db = DatabaseManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization and table creation"""
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='test_results'
        """)
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()
    
    def test_save_result(self):
        """Test saving test results"""
        result = TestResult(
            test_id="test_001",
            model_name="test_model",
            response="Test response",
            response_time=1.5,
            timestamp="2024-01-01T00:00:00",
            scores={"overall_quality": 0.85},
            metadata={"category": "test"}
        )
        
        # Save result
        self.db.save_result(result)
        
        # Retrieve and verify
        results = self.db.get_results()
        self.assertEqual(len(results), 1)
        
        saved_result = results[0]
        self.assertEqual(saved_result['test_id'], "test_001")
        self.assertEqual(saved_result['model_name'], "test_model")
        self.assertEqual(saved_result['response'], "Test response")
    
    def test_get_results_filtering(self):
        """Test result filtering"""
        # Save multiple results
        for i in range(3):
            result = TestResult(
                test_id=f"test_{i:03d}",
                model_name="test_model" if i < 2 else "other_model",
                response=f"Response {i}",
                response_time=1.0 + i,
                timestamp="2024-01-01T00:00:00",
                scores={"overall_quality": 0.8},
                metadata={"category": "test"}
            )
            self.db.save_result(result)
        
        # Test filtering by model
        model_results = self.db.get_results(model_name="test_model")
        self.assertEqual(len(model_results), 2)
        
        # Test filtering by test_id
        specific_result = self.db.get_results(test_id="test_001")
        self.assertEqual(len(specific_result), 1)
        self.assertEqual(specific_result[0]['test_id'], "test_001")


class TestConfig(unittest.TestCase):
    """Test cases for configuration management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        test_config = {
            "models": {
                "ollama": {
                    "timeout": 60,
                    "max_retries": 2
                }
            },
            "testing": {
                "max_workers": 2
            }
        }
        json.dump(test_config, self.temp_config)
        self.temp_config.close()
        
        self.config = Config(self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        os.unlink(self.temp_config.name)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertEqual(self.config.get("models.ollama.timeout"), 60)
        self.assertEqual(self.config.get("models.ollama.max_retries"), 2)
        self.assertEqual(self.config.get("testing.max_workers"), 2)
    
    def test_config_defaults(self):
        """Test default configuration values"""
        # Non-existent key should return default
        self.assertEqual(self.config.get("non.existent.key", "default"), "default")
        
        # Existing keys from default config
        self.assertIsNotNone(self.config.get("models.ollama.default_models"))
    
    def test_config_modification(self):
        """Test configuration modification"""
        self.config.set("testing.new_setting", "test_value")
        self.assertEqual(self.config.get("testing.new_setting"), "test_value")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = LLMTestFramework(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('ollama.chat')
    def test_complete_test_workflow(self, mock_chat):
        """Test complete testing workflow"""
        # Mock Ollama response
        mock_chat.return_value = {
            'message': {'content': 'The answer is 4'}
        }
        
        # Add model
        self.framework.add_ollama_model("test_model")
        
        # Create test cases
        test_cases = [
            TestCase(
                id="math_001",
                category="math",
                prompt="What is 2 + 2?",
                expected_type="numeric"
            ),
            TestCase(
                id="code_001",
                category="programming",
                prompt="Write a function to add two numbers",
                expected_type="code"
            )
        ]
        
        # Run tests
        results = self.framework.run_custom_tests(test_cases)
        
        # Verify results
        self.assertIn("test_model", results)
        self.assertEqual(len(results["test_model"]), 2)
        
        # Check that results were saved to database
        db_results = self.framework.db.get_results()
        self.assertEqual(len(db_results), 2)
    
    def test_report_generation(self):
        """Test report generation"""
        # Create mock results
        mock_results = {
            "test_model": [
                TestResult(
                    test_id="test_001",
                    model_name="test_model",
                    response="Test response",
                    response_time=1.5,
                    timestamp="2024-01-01T00:00:00",
                    scores={"overall_quality": 0.85}
                )
            ]
        }
        
        # Generate report
        report = self.framework.generate_report(mock_results)
        
        # Verify report content
        self.assertIn("LLM Testing Report", report)
        self.assertIn("test_model", report)
        self.assertIn("1.50s", report)  # Response time
        self.assertIn("0.85", report)   # Quality score


if __name__ == '__main__':
    # Create test results directory
    Path("test_results").mkdir(exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)