# 🤖 Python AI Framework Comparison

This repository contains a modular, extensible framework for testing and comparing Large Language Models (LLMs) such as `llama3`, `mistral`, and more. It allows for standardized prompt evaluation, automated scoring, database logging, and visual reporting.

> Built to help engineers and researchers assess speed, quality, and consistency of AI models with minimal setup.

---

## 📌 Table of Contents

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

- 🔌 Plugin-style architecture for model providers (Ollama, OpenAI, etc.)
- 🧪 Runs categorized prompt tests across any model
- 🧠 Intelligent response scoring:
  - Length analysis
  - Factual confidence
  - Code quality
  - Creativity detection
- 🗂️ Built-in test cases across math, reasoning, coding, factual, and creative categories
- 📊 HTML + CSV reports with comparative insights
- 🗃️ SQLite storage of all results for future querying

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
- [ ] OpenAI & Claude support
- [ ] Parallel test execution (multithreaded)
- [ ] CLI test filtering (by category, model)
- [ ] Graphs & charts in reports (Plotly or Chart.js)

---

## 👨‍💻 Author

**Gaurav Singh**

- 🌍 Portfolio: [gauravsingh-info.netlify.app](https://gauravsingh-info.netlify.app)
- 💼 LinkedIn: [linkedin.com/in/gaurav-singh27](https://www.linkedin.com/in/gaurav-singh27/)
- 📧 Email: [gaurav10690@gmail.com](mailto:gaurav10690@gmail.com)

> ✨ If you find this project useful, please consider giving it a ⭐ on GitHub!
