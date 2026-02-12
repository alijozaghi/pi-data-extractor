<div align="center">

# 📊 PI Data Extractor

**Enterprise Streamlit App for PI Time-Series Retrieval, Cleaning, and Export**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ Overview

**PI Data Extractor** is an internal Streamlit application that retrieves interpolated time-series data from **OSIsoft PI** using an Excel-based tag list, applies **data-quality cleaning**, and exports analysis-ready datasets to a professionally formatted Excel file.

> ✅ Designed for **internal environments** where PI connectivity and authentication are available.

---

## ✅ Key Features

### 📁 Flexible Tag List Input (Dynamic Metadata)
Upload an Excel file with a required **SCADA TAG** column plus **any number of metadata columns**.  
All metadata columns are preserved and written as **multi-level headers** in the exported Excel file.

### 🔁 Two Export Modes (No Re-Query Needed)
After one PI query, you can switch exports instantly:

- **Raw Data** — values exactly as returned by PI  
- **Clean Data** — structured rules applied:
  - status strings → `0/1` (e.g., ON/OFF, ACTIVE/INACTIVE, OPEN/CLOSED)
  - non-numeric → `0`
  - negative values → clipped to `0`
  - known “bad” tokens handled consistently

### 🧠 Controlled Query Execution
Changing **time range / interval / timezone** does **not** auto-query PI.  
Data retrieval happens only when you click:

**Run / Refresh PI Query**

### 📦 Professional Excel Export
- multi-level headers (from metadata)
- frozen panes
- numeric format to **3 decimals**
- clean structure for downstream engineering/analytics workflows

### 🧪 Diagnostics & Transparency
- tags that fail retrieval are listed with error messages
- time-series plots can highlight:
  - raw negatives
  - raw missing/bad tokens

---

## 📄 Input File Format

Your uploaded Excel must include one tag column with any of these names:

- `SCADA TAG` (**recommended**)
- `SCADA_TAG`
- `TAG`
- `PI TAG`

All other columns are treated as metadata (can be blank).

### Example

| Facility | Unit   | SCADA TAG     |
|---------|--------|---------------|
| Plant A | Pump 1 | TAG.NAME.001  |
| Plant A | Pump 2 | TAG.NAME.002  |

---

## 🧱 Tech Stack

- **UI**: Streamlit  
- **PI Access**: PIconnect  
- **Charts**: Altair  
- **Export**: Pandas + XlsxWriter  
- **Timezone**: pytz  

---

## ▶️ Run Locally (Recommended for PI Environments)

> PI connectivity typically requires internal network access and/or domain authentication.

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
