# Aegix

# Ollama-Based Jailbreak Detection & Evaluation Framework

This project provides a Python framework for detecting potential jailbreak attempts or harmful content in text inputs using a combination of regular expressions and a locally running Ollama language model (e.g., DeepSeek). It also includes tools for evaluating the detector's performance using academic metrics.

## Features

* **Hybrid Detection:** Combines fast regex-based static rule matching with more nuanced LLM-based semantic analysis.
* **Multiple Detection Paths:** Allows testing and comparison of:
    * Regex-only detection
    * Model-only detection
    * Full pipeline (Regex -> Model)
* **Ollama Integration:** Easily configurable to use different models available through a local Ollama instance.
* **Configurable:** Key parameters (API endpoints, model names, rules, file paths, sample sizes) are managed in a central `config.py` file.
* **Dataset Handling:** Loads attack and benign samples from CSV files for testing.
* **Stratified Sampling:** Implements logic to sample from datasets for balanced testing.
* **Performance Evaluation:**
    * Calculates standard metrics: Accuracy, Precision, Recall, F1-Score, FPR, FNR.
    * Tracks processing time and block rates for different detection paths.
* **Live Reporting:** Displays detection results for each sample in real-time.
* **Modular Design:** Code is organized into logical modules for better readability and maintainability.

## Directory Structure

```
your_project_root/
├── main.py                     # Main script to run experiments
├── ollama_client.py            # Ollama API client
├── detectors.py                # Jailbreak detection logic (Simple & Enhanced)
├── evaluation.py               # Academic evaluation and experiment recording
├── utils.py                    # Utility functions (data loading, table display)
├── config.py                   # Configuration (model names, paths, rules, etc.)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Directory for your CSV datasets (create this directory)
 ├── HEx-PHI.csv             # Example attack dataset
 ├── JailBreak.csv           # Example attack dataset
 └── harmless_samples.csv    # Example benign dataset
└── origin/                   # Original code
 └──  data.py             # Original code, with everthing in one file
```


## Prerequisites

* **Python 3.7+**
* **Ollama:**
    * Ensure Ollama is installed and running. You can find installation instructions at [https://ollama.com/](https://ollama.com/).
    * Pull the language model you intend to use (e.g., `deepseek-r1:7b` which is the default in `config.py`):
        ```bash
        ollama pull deepseek-r1:7b
        ```
        (If you change `SAFETY_MODEL_NAME` in `config.py`, pull that model instead.)

## Setup & Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    * Create a `data/` directory in the project root.
    * Place your attack and harmless sample CSV files into the `data/` directory.
    * The CSV files should have at least two columns, with the header in the first row. The text content to be analyzed is expected to be in the **second column** (index 1).
    * Update `ATTACK_FILES` and `HARMLESS_FILES` lists in `config.py` if your filenames are different.

## Configuration

Most settings can be adjusted in `config.py`:

* `OLLAMA_BASE_URL`: URL for your Ollama API.
* `OLLAMA_TIMEOUT`: Timeout for API requests.
* `SAFETY_MODEL_NAME`: The Ollama model to use for safety checks.
* `STATIC_RULES`: List of `re.compile()` objects for regex detection.
* `DATA_DIR`: Path to the directory containing your datasets.
* `ATTACK_FILES`, `HARMLESS_FILES`: Lists of filenames for attack and benign datasets.
* `DEFAULT_SAMPLE_SIZE`: Number of samples to draw from external datasets for testing.

## Usage

To run the experiment, execute the `main.py` script from the project root directory:

```bash
python main.py
