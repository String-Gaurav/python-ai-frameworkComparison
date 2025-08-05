"""
Parallel Test Execution for LLM Framework
Implements concurrent testing to improve performance
"""

import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any, Callable, Optional
import logging
from dataclasses import dataclass

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ParallelTestConfig:
    """Configuration for parallel test execution"""
    max_workers: int = 3
    timeout_per_test: int = 120
    retry_failed_tests: bool = True
    batch_size: int = 10


class ParallelTestRunner:
    """Manages parallel execution of LLM tests"""
    
    def __init__(self, config: ParallelTestConfig = None):
        self.config = config or ParallelTestConfig()
        self.config.max_workers = min(
            self.config.max_workers, 
            config.get("testing.max_workers", 3)
        )
    
    def run_tests_parallel(
        self, 
        test_functions: List[Callable],
        test_args: List[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple test functions in parallel using ThreadPoolExecutor
        
        Args:
            test_functions: List of test functions to execute
            test_args: List of argument tuples for each test function
            
        Returns:
            List of test results
        """
        if test_args is None:
            test_args = [() for _ in test_functions]
        
        if len(test_functions) != len(test_args):
            raise ValueError("Number of test functions and arguments must match")
        
        results = []
        failed_tests = []
        
        logger.info(f"Starting parallel execution of {len(test_functions)} tests with {self.config.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._run_single_test, func, args): (i, func, args)
                for i, (func, args) in enumerate(zip(test_functions, test_args))
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_test, timeout=self.config.timeout_per_test * 2):
                test_index, test_func, test_args = future_to_test[future]
                
                try:
                    result = future.result(timeout=self.config.timeout_per_test)
                    result['test_index'] = test_index
                    results.append(result)
                    logger.debug(f"Test {test_index} completed successfully")
                    
                except Exception as e:
                    error_result = {
                        'test_index': test_index,
                        'error': str(e),
                        'status': 'failed',
                        'test_function': test_func.__name__ if hasattr(test_func, '__name__') else str(test_func)
                    }
                    failed_tests.append((test_index, test_func, test_args, e))
                    results.append(error_result)
                    logger.error(f"Test {test_index} failed: {e}")
        
        # Retry failed tests if configured
        if self.config.retry_failed_tests and failed_tests:
            logger.info(f"Retrying {len(failed_tests)} failed tests")
            retry_results = self._retry_failed_tests(failed_tests)
            
            # Update results with retry outcomes
            for retry_result in retry_results:
                for i, result in enumerate(results):
                    if result.get('test_index') == retry_result.get('test_index'):
                        results[i] = retry_result
                        break
        
        # Sort results by test_index to maintain order
        results.sort(key=lambda x: x.get('test_index', 0))
        
        logger.info(f"Parallel execution completed. {len([r for r in results if r.get('status') != 'failed'])} successful, {len([r for r in results if r.get('status') == 'failed'])} failed")
        
        return results
    
    def _run_single_test(self, test_func: Callable, args: tuple) -> Dict[str, Any]:
        """Run a single test function with error handling"""
        start_time = time.time()
        
        try:
            result = test_func(*args)
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result['execution_time'] = execution_time
                result['status'] = 'success'
                return result
            else:
                return {
                    'result': result,
                    'execution_time': execution_time,
                    'status': 'success'
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Test function {test_func.__name__ if hasattr(test_func, '__name__') else 'unknown'} failed: {e}")
            raise
    
    def _retry_failed_tests(self, failed_tests: List[tuple]) -> List[Dict[str, Any]]:
        """Retry failed tests with exponential backoff"""
        retry_results = []
        
        for test_index, test_func, test_args, original_error in failed_tests:
            try:
                logger.info(f"Retrying test {test_index}")
                time.sleep(1)  # Brief delay before retry
                
                result = self._run_single_test(test_func, test_args)
                result['test_index'] = test_index
                result['retried'] = True
                result['original_error'] = str(original_error)
                retry_results.append(result)
                
            except Exception as retry_error:
                retry_results.append({
                    'test_index': test_index,
                    'error': str(retry_error),
                    'original_error': str(original_error),
                    'status': 'failed_retry',
                    'retried': True
                })
        
        return retry_results
    
    async def run_tests_async(
        self, 
        async_test_functions: List[Callable],
        test_args: List[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Run async test functions concurrently
        
        Args:
            async_test_functions: List of async test functions
            test_args: List of argument tuples for each test function
            
        Returns:
            List of test results
        """
        if test_args is None:
            test_args = [() for _ in async_test_functions]
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def run_with_semaphore(func, args):
            async with semaphore:
                return await self._run_single_async_test(func, args)
        
        tasks = [
            run_with_semaphore(func, args)
            for func, args in zip(async_test_functions, test_args)
        ]
        
        logger.info(f"Starting async execution of {len(tasks)} tests")
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_per_test * 2
            )
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'test_index': i,
                        'error': str(result),
                        'status': 'failed'
                    })
                else:
                    result['test_index'] = i
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error("Async test execution timed out")
            return [{'error': 'Timeout', 'status': 'failed', 'test_index': i} for i in range(len(async_test_functions))]
    
    async def _run_single_async_test(self, test_func: Callable, args: tuple) -> Dict[str, Any]:
        """Run a single async test function"""
        start_time = time.time()
        
        try:
            result = await test_func(*args)
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result['execution_time'] = execution_time
                result['status'] = 'success'
                return result
            else:
                return {
                    'result': result,
                    'execution_time': execution_time,
                    'status': 'success'
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Async test function failed: {e}")
            return {
                'error': str(e),
                'execution_time': execution_time,
                'status': 'failed'
            }


class BatchProcessor:
    """Process large numbers of tests in batches"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.runner = ParallelTestRunner()
    
    def process_in_batches(
        self, 
        test_functions: List[Callable],
        test_args: List[tuple] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process tests in batches to avoid overwhelming the system
        
        Args:
            test_functions: List of test functions
            test_args: List of argument tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            Combined results from all batches
        """
        if test_args is None:
            test_args = [() for _ in test_functions]
        
        total_tests = len(test_functions)
        all_results = []
        
        logger.info(f"Processing {total_tests} tests in batches of {self.batch_size}")
        
        for i in range(0, total_tests, self.batch_size):
            batch_end = min(i + self.batch_size, total_tests)
            batch_functions = test_functions[i:batch_end]
            batch_args = test_args[i:batch_end]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_tests + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self.runner.run_tests_parallel(batch_functions, batch_args)
            
            # Adjust test indices for global ordering
            for result in batch_results:
                if 'test_index' in result:
                    result['test_index'] += i
            
            all_results.extend(batch_results)
            
            if progress_callback:
                progress_callback(batch_end, total_tests)
            
            # Brief pause between batches to prevent overwhelming
            time.sleep(0.5)
        
        logger.info(f"Batch processing completed. Total results: {len(all_results)}")
        return all_results


# Convenience functions
def run_parallel_tests(test_functions: List[Callable], max_workers: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to run tests in parallel"""
    config = ParallelTestConfig(max_workers=max_workers)
    runner = ParallelTestRunner(config)
    return runner.run_tests_parallel(test_functions)


async def run_async_tests(async_test_functions: List[Callable], max_workers: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to run async tests"""
    config = ParallelTestConfig(max_workers=max_workers)
    runner = ParallelTestRunner(config)
    return await runner.run_tests_async(async_test_functions)