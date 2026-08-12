# AlphaGenome Mutagenesis Workflows

This repository provides two interactive workflows for saturation mutagenesis with AlphaGenome:

- `gene_mutagenisis.py`: Gene/TSS-centered workflow (promoter-style runs).
- `interval_mutagenisis.py`: Interval/enhancer-centered workflow.

These scripts call the shared pipeline internally. Most users only need to run one of the two scripts above.

## Requirements

- Python 3.11 (recommended)
- Internet access (AlphaGenome API + sequence/annotation lookups)
- AlphaGenome API key

## 1) Install Python

### macOS

Option 1: download Python 3.11 from https://www.python.org/downloads/

Option 2: install with Homebrew.

If Homebrew is not installed, install it first:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python 3.11:

```bash
brew install python@3.11
```

Verify installation:

```bash
python3.11 --version
```

### Windows

Download Python 3.11 from https://www.python.org/downloads/windows/

During installation:

- check `Add python.exe to PATH`
- choose Python 3.11

Verify installation in PowerShell:

```powershell
py -3.11 --version
```

### Linux

Install Python 3.11 using your package manager.

Ubuntu/Debian example:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

Verify installation:

```bash
python3.11 --version
```

## 2) Clone and Set Up

### macOS/Linux

```bash
git clone https://github.com/Kaploc23/AlphaGenome.git
cd AlphaGenome

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Kaploc23/AlphaGenome.git
cd AlphaGenome

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install Dependencies

Install the Python packages required by the workflows:

### macOS/Linux

```bash
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
pip install -r requirements.txt
```

This installs the dependencies listed in [requirements.txt](requirements.txt), including AlphaGenome, pandas, numpy, matplotlib, scipy, tqdm, Biopython, and openpyxl.

## 4) Configure API Key

### macOS/Linux

```bash
export ALPHAGENOME_API_KEY="YOUR_API_KEY"
```

### Windows (PowerShell)

```powershell
setx ALPHAGENOME_API_KEY "YOUR_API_KEY"
```

Note: on Windows, open a new shell after `setx` so the variable is available.

## 5) Run a Workflow

### Gene/TSS workflow

```bash
python gene_mutagenisis.py
```

### Interval/enhancer workflow

```bash
python interval_mutagenisis.py
```
If "python" is not working try "python3"
## Outputs

Outputs are written under the repository `Outputs/` directory, including:

- scored CSVs
- metadata summary CSVs
- plots
- FASTA context/window files

## Notes on Filtering

- The scripts support ontology-based filtering.
- Some ontology IDs may not be supported by AlphaGenome request-time filters.
- Current workflow logic includes fallback behavior so unsupported or overly strict filters do not hard-crash long runs.

## Troubleshooting

### `Unsupported ontology` error

Use a different ontology term or run with a broader filter. The workflow can fall back when request-time filtering is rejected.

### `No requested outputs were returned by AlphaGenome`

This typically means filters were too strict for available tracks. The pipeline now retries with relaxed post-filter constraints.

### Network / proxy errors (for example `403 Forbidden` tunnel)

Your environment is blocking outbound requests. Allow HTTPS access to required services (AlphaGenome and sequence providers).

### Matplotlib cache warning

If you see a non-writable cache warning, set a writable config dir:

```bash
export MPLCONFIGDIR="$HOME/.config/matplotlib"
```

## Reproducibility Tips

- Pin dependency versions in `requirements.txt` for long-term reproducibility.
- Save command logs and generated metadata summaries with each run.
- Rotate API keys if they are ever exposed in terminal history or logs.
