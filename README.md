# Local GenAI Data Quality Profiler

A fully offline, AI-powered data quality tool for financial datasets. Upload a CSV, get instant programmatic findings, AI-generated insights, testable validation rules, and a downloadable Excel report — all without sending a single byte to an external API.

Built as a portfolio project for a data quality engineering role, demonstrating how generative AI can accelerate data quality assessments in financial institutions.

---

## Why this exists

Financial institutions deal with enormous amounts of data that must meet strict quality standards for regulatory compliance. Manual data quality assessment is slow, inconsistent, and doesn't scale. This tool replicates the core idea behind tools like Deloitte's dQube: use AI to generate and apply data quality rules automatically, surface problems fast, and give a human the information they need to decide what to do.

**Key design principle: the tool is read-only.** It never modifies the data. In regulated financial environments, you cannot silently correct source data without an audit trail and client sign-off. The tool surfaces and quantifies problems. Remediation decisions always stay with the domain expert.

**Why Ollama instead of a cloud API:** In financial institutions, client data often cannot leave the building. Using a local model via Ollama means the tool works fully air-gapped. No data is ever sent to OpenAI, Anthropic, or any external service.

---

## Features

### Step 1 — Instant Audit (no AI, instant results)
Runs programmatic checks on the full dataset the moment a file is uploaded. No waiting for a model. Findings are colour-coded by severity:

- **Critical** (red): issues that will break downstream analysis
- **Warning** (orange): issues that may skew results
- **Info** (blue): observations worth investigating

Checks include:
- Null value detection per column with percentages
- Duplicate row detection
- Extreme outliers using 3x IQR method
- Negative values in columns that should be positive
- Zero-heavy columns
- Constant columns (all values identical)
- Inconsistent capitalisation in categorical columns
- Columns where every value is unique (potential ID columns)

### Step 2 — AI Analysis
Sends a compressed profile and a sample of suspicious rows to a local Ollama model. The model returns a prioritised bullet list of data quality issues.

Two modes:
- **Sample (fast)**: extracts up to 100 suspicious rows, sends them with statistical summary. Typical response time 15-30 seconds.
- **Full dataset (batched)**: extracts up to 500 suspicious rows, processes in parallel batches of configurable size, consolidates findings. Slower but more thorough.

The tool does not send all rows to the model. It pre-filters using statistical methods (outliers, nulls, duplicates, rare categorical values) so the model only reasons over rows that are actually worth investigating. This keeps token count low, response time fast, and is the correct approach for large datasets.

### Step 3 — Rule Validation
Ollama generates testable validation rules based on the dataset profile. Each rule is a Python/pandas expression that flags bad rows. The tool then runs every rule against the full dataset and returns exact row counts.

Example rule:
```
Column: loan_amount
Rule: loan amount must be positive
Filter: df['loan_amount'] <= 0
Result: 47 rows fail (0.3%)
```

You can click into any failing rule to inspect the actual rows. This closes the loop between AI suggestion and hard evidence: the model suggests what should be true, the code proves how many rows violate it.

### Step 4 — Export Report
Generates a formatted multi-sheet Excel report containing:
- **Overview**: file metadata and full instant audit summary with colour-coded severity
- **Instant Audit**: full programmatic findings table
- **AI Analysis**: complete AI-generated findings as text
- **Rule Validation**: all rules with pass/fail status and row counts
- **Failing rows sheets**: one sheet per failed rule showing the actual offending rows (up to 50 per rule)

The report button automatically runs any steps that haven't been run yet, so you always get a complete report.

### Step 5 — Follow-up Chat
After running the AI analysis, a chat interface opens in the sidebar. The full conversation history is maintained so Ollama has context for every follow-up question. Useful for asking why a specific issue was flagged, what it means in a financial context, or what remediation options exist.

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| UI | Streamlit |
| Local LLM | Ollama (gemma3:4b) |
| Data processing | pandas |
| SQL engine | DuckDB |
| Report generation | openpyxl |
| Template rendering | Jinja2 |
| Parallel processing | concurrent.futures (ThreadPoolExecutor) |

---

## Project structure

```
local-genai-data-quality/
├── src/
│   ├── profiler.py       # all logic: checks, AI calls, rule validation, report generation
│   └── app.py            # Streamlit UI
├── Datasets/             # sample datasets for testing
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed
- The `gemma3:4b` model pulled

### Installation

```bash
# clone the repo
git clone https://github.com/yourusername/local-genai-data-quality.git
cd local-genai-data-quality

# install dependencies
pip3 install -r requirements.txt

# pull the model
ollama pull gemma3:4b
```

### Running the app

```bash
python3 -m streamlit run src/app.py
```

The app automatically kills any existing Ollama instance and restarts it with `OLLAMA_NUM_PARALLEL=8` for parallel batch processing. No separate terminal needed.

### CLI usage

The profiler can also be run from the command line without the UI:

```bash
python3 src/profiler.py Datasets/titanic.csv
```

This prints the instant audit table and streams the AI analysis directly to the terminal.

---

## Sample datasets

The following datasets work well for testing and demonstrate different types of data quality issues:

| Dataset | Source | Notable issues |
|---|---|---|
| Loan Approval | [Kaggle](https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Loan-Approval-Prediction.csv) | Nulls, mixed types |
| Insurance Charges | [pycaret](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv) | Duplicate rows, string-coded booleans |
| Bank Marketing | [pandas-tutorial](https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/bank.csv) | "unknown" as a category value |
| S&P 500 Financials | [datasets](https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/main/data/constituents-financials.csv) | Nulls in financial ratios, negative EBITDA |
| Titanic | [datasciencedojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) | Classic nulls + mixed types, good sanity check |
| Employee Attrition | [pycaret](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/employee.csv) | Categorical consistency |
| France E-commerce | [pycaret](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/france.csv) | Negative quantities, duplicates |

Download directly:
```bash
curl -o Datasets/insurance.csv https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv
curl -o Datasets/titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

---

## Extending the tool

Adding a new programmatic check is one function and one line:

```python
# in profiler.py, add a function
def check_future_dates(df):
    result = {}
    for col in df.select_dtypes(include="datetime").columns:
        future = (df[col] > pd.Timestamp.now()).sum()
        if future > 0:
            result[col] = {"future_dates": int(future)}
    return result

# then add it to the CHECKS list
CHECKS = [
    ...
    ("future_dates", check_future_dates),
]
```

The new check automatically appears in the instant audit, gets included in the profile sent to Ollama, and shows up in the exported report.

---

## Architecture decisions

**Why pre-filter before sending to the model?**
Sending all rows of a large dataset to a local LLM is impractical: too slow, hits context limits, and most rows are fine anyway. The tool uses statistical methods (IQR outliers, null flags, rare value detection) to identify the rows most likely to contain problems. The model then reasons over that subset plus a full statistical summary. For a 15,000 row dataset this reduces the model input from 15,000 rows to typically 50-200 rows, while still giving the model full statistical awareness of the whole dataset.

**Why parallel batch processing?**
For large suspicious row sets, the tool splits them into batches and processes up to 8 simultaneously using `ThreadPoolExecutor`. A final consolidation call merges all batch findings into a single prioritised list. Ollama must be started with `OLLAMA_NUM_PARALLEL=8` to actually process requests in parallel, which the app handles automatically on startup.

**Why is the tool read-only?**
What looks like a flaw might be valid data. A negative loan amount could be a repayment. An age of 150 could be a coded missing value the source system uses intentionally. Only the domain expert has the business context to decide. In regulated financial environments, modifying source data without an audit trail is itself a compliance violation. The tool's job is to surface and quantify, not to fix.

---

## Requirements

```
pandas
duckdb
ollama
jinja2
streamlit
openpyxl
```

---

## License

MIT