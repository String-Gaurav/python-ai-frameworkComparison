# 🤖 Python AI Framework Comparison

This repository contains a modular, extensible framework for testing and comparing Large Language Models (LLMs) such as `llama3`, `mistral`, and more. It allows for standardized prompt evaluation, automated scoring, database logging, and visual reporting.

> Built to help engineers and researchers assess speed, quality, and consistency of AI models with minimal setup.

---

## 📌 Table of Contents

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage Example](#usage-example)
- [Scoring Metrics](#scoring-metrics)
- [Test Categories](#test-categories)
- [Sample Report](#sample-report)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Author](#author)


---

## 🚀 Features

- 🔌 **Multi-Provider Support**: Ollama, OpenAI, Anthropic with automatic fallback
- 🧪 **Comprehensive Testing**: Categorized prompt tests with customizable test suites
- 🧠 **Intelligent Scoring**: Multi-dimensional analysis (length, confidence, creativity, code quality)
- ⚡ **Parallel Execution**: Multi-threaded testing for improved performance
- 📊 **Rich Reporting**: Interactive HTML reports with Plotly charts and visualizations
- 🗃️ **Persistent Storage**: SQLite database with advanced querying and filtering
- ⚙️ **Configuration Management**: Flexible JSON-based configuration with environment variable support
- 🛠️ **Developer Tools**: Comprehensive test suite, code quality tools, CI/CD pipeline
- 📈 **Performance Benchmarking**: Built-in benchmarking tools with statistical analysis
- 🖥️ **Enhanced CLI**: Full-featured command-line interface with subcommands and filtering

---

## 🗂️ Project Structure

```
python-ai-frameworkComparison/
├── llm_test_framework.py         # Main testing framework
├── export_my_results.py          # CSV/HTML export script
├── sampleTest.py                 # Example test runner
├── test_runner.py                # CLI execution
├── testingLLM.py                 # Integration runner
├── .gitignore
├── requirements.txt
└── test_results/
    ├── llm_test_results.db
    ├── model_comparison_results.json
    ├── llm_test_results_*.csv
    └── llm_test_summary_*.html
```

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/String-Gaurav/python-ai-frameworkComparison.git
cd python-ai-frameworkComparison
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Framework

Using the enhanced CLI:

```bash
# Basic usage
python cli.py run --models llama3.2 mistral:7b --categories math programming

# With OpenAI/Anthropic models (requires API keys)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
python cli.py run --models gpt-3.5-turbo claude-3-haiku llama3.2

# Parallel execution
python cli.py run --models llama3.2 mistral:7b --parallel --workers 4

# Custom test file
python cli.py run --models llama3.2 --test-file custom_tests.json

# Benchmark models
python cli.py benchmark --models llama3.2 mistral:7b --iterations 5

# Generate enhanced reports
python cli.py report --format enhanced --output results.html

# Check system status
python cli.py status --check-models

# List available models
python cli.py list-models --provider ollama
```

Legacy interface (still supported):
```bash
python test_runner.py
```

---

## 💡 Usage Example

```python
from llm_test_framework import LLMTestFramework

framework = LLMTestFramework()
framework.add_ollama_model("llama3.2:latest")

# Run math category test
results = framework.run_test_suite(categories=["math"])

# Generate reports
framework.generate_report(results)
framework.export_results_csv()
```

---

## 📊 Scoring Metrics

Each model response is scored based on:

| Metric           | Applied To     | Description                               |
|------------------|----------------|-------------------------------------------|
| `length_score`   | All types      | Penalizes too short/long answers          |
| `code_quality`   | Coding         | Keyword and syntax indicator check        |
| `confidence`     | Factual        | Favors confident vs uncertain phrases     |
| `creativity`     | Creative       | Looks for storytelling and variation      |
| `math_content`   | Math           | Checks for numeric/symbolic content       |
| `overall_quality`| All types      | Combined result of above per type         |

---

## 🧪 Test Categories

Built-in test suite covers:

- 🧮 `math` – numerical logic
- 💻 `code` – Python generation
- 💡 `factual` – knowledge prompts
- 🧠 `reasoning` – logical deduction
- ✍️ `creative` – storytelling, poetry

Each test case includes:
- ID, prompt, type, expected output, category, difficulty, and tags

---

## 📈 Sample Report

Example output:

| Model           | Response Time | Quality Score | Tests Run |
|-----------------|---------------|----------------|-----------|
| llama3.2:latest | 4.04s         | 0.83           | 14        |
| mistral:7b      | 7.10s         | 0.83           | 14        |

✅ **Key Insight**: llama3.2:latest maintained equal quality with 76% faster response time vs mistral:7b.

📁 Reports are saved under `/test_results/` in `.csv`, `.json`, and `.html` formats.

<img width="850" height="923" alt="image" src="https://github.com/user-attachments/assets/35c832d1-3a59-4dc7-ace1-e17bdbb4c3ce" />


---

## 📦 Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Content of `requirements.txt`:

```txt
ollama
pandas
jinja2
beautifulsoup4
```

---

## 🧭 Roadmap

- [x] SQLite-backed persistent storage
- [x] Visual HTML report with comparison
- [x] OpenAI & Claude support
- [x] Parallel test execution (multithreaded)
- [x] CLI test filtering (by category, model)
- [x] Graphs & charts in reports (Plotly)
- [x] Enhanced configuration management
- [x] Comprehensive testing suite
- [x] CI/CD pipeline with GitHub Actions
- [x] Code quality tools (Black, Flake8, MyPy)
- [x] Performance benchmarking
- [ ] Real-time test execution monitoring
- [ ] Docker containerization
- [ ] Web dashboard interface
- [ ] Custom scoring algorithms
- [ ] Model fine-tuning integration

---

## 👨‍💻 Author

**Gaurav Singh**

- 🌍 Portfolio: [gauravsingh-info.netlify.app](https://gauravsingh-info.netlify.app)
- 💼 LinkedIn: [linkedin.com/in/gaurav-singh27](https://www.linkedin.com/in/gaurav-singh27/)
- 📧 Email: [gaurav10690@gmail.com](mailto:gaurav10690@gmail.com)

> ✨ If you find this project useful, please consider giving it a ⭐ on GitHub!
