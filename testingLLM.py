import ollama
import time
import json


def test_model_comparison():
    # Your available models
    models = ["llama3.2:latest", "mistral:7b"]

    # Test prompts covering different capabilities
    test_prompts = [
        {
            "category": "General Reasoning",
            "prompt": "If a train leaves New York at 3 PM traveling 60 mph, and another leaves Boston at 4 PM traveling 80 mph, when will they meet? The cities are 200 miles apart."
        },
        {
            "category": "Code Generation",
            "prompt": "Write a Python function that finds the longest word in a sentence and returns both the word and its length."
        },
        {
            "category": "Creative Writing",
            "prompt": "Write a short story (3-4 sentences) about a robot learning to paint."
        },
        {
            "category": "Factual Knowledge",
            "prompt": "Explain the difference between machine learning and deep learning in 2-3 sentences."
        },
        {
            "category": "Problem Solving",
            "prompt": "I have 100 apples. I give away 30% to my neighbor and eat 15 myself. How many apples do I have left?"
        }
    ]

    results = {}

    for model in models:
        print(f"\n{'=' * 50}")
        print(f"TESTING MODEL: {model}")
        print(f"{'=' * 50}")

        model_results = {}

        for test in test_prompts:
            category = test["category"]
            prompt = test["prompt"]

            print(f"\n--- {category} ---")
            print(f"Prompt: {prompt}")

            try:
                start_time = time.time()
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                end_time = time.time()

                response_time = end_time - start_time
                response_text = response['message']['content']

                print(f"Response Time: {response_time:.2f} seconds")
                print(f"Response: {response_text}")

                model_results[category] = {
                    "response": response_text,
                    "response_time": response_time,
                    "prompt": prompt
                }

            except Exception as e:
                print(f"Error with {model}: {e}")
                model_results[category] = {"error": str(e)}

        results[model] = model_results

        # Brief pause between models
        time.sleep(1)

    return results


def analyze_results(results):
    print(f"\n{'=' * 60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'=' * 60}")

    # Response time analysis
    print("\n--- Response Time Comparison ---")
    for model in results:
        times = [result["response_time"] for result in results[model].values()
                 if "response_time" in result]
        if times:
            avg_time = sum(times) / len(times)
            print(f"{model}: {avg_time:.2f}s average")

    # Category-wise comparison
    print("\n--- Category-wise Performance ---")
    categories = set()
    for model_results in results.values():
        categories.update(model_results.keys())

    for category in categories:
        print(f"\n{category}:")
        for model in results:
            if category in results[model] and "response" in results[model][category]:
                response = results[model][category]["response"]
                time_taken = results[model][category]["response_time"]
                print(f"  {model} ({time_taken:.1f}s): {response[:100]}...")


def save_results(results, filename="model_comparison_results.json"):
    """Save results to a JSON file for later analysis"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def quick_test():
    """Quick test to verify all models are working"""
    models = ["llama3.2:latest", "mistral:7b"]

    print("Quick verification test...")
    for model in models:
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': 'Say hello and your model name'}]
            )
            print(f"✓ {model}: Working")
        except Exception as e:
            print(f"✗ {model}: Error - {e}")


if __name__ == "__main__":
    print("AI Model Comparison Testing")
    print("This will test multiple models across different capabilities...")

    # Quick verification first
    quick_test()

    # Ask user if they want to proceed with full test
    proceed = input("\nProceed with full comparison test? (y/n): ").lower().strip()

    if proceed == 'y':
        results = test_model_comparison()
        analyze_results(results)
        save_results(results)

        print("\n" + "=" * 60)
        print("TESTING COMPLETE!")
        print("Check the saved JSON file for detailed results.")
        print("You can use this data to decide which model works best for different tasks.")
    else:
        print("Test cancelled.")