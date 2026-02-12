PI Data Extractor
Enterprise Streamlit Application for PI Time-Series Retrieval & Data Quality Processing

──────────────────────────────────────────────────────────────────────────────

OVERVIEW

PI Data Extractor is an internal Streamlit application designed to retrieve
interpolated time-series data from OSIsoft PI using a structured Excel-based
tag list. The application supports controlled data retrieval, automated data
quality cleaning, and professionally formatted Excel exports suitable for
engineering and operational workflows.

This tool is intended for internal environments where direct PI connectivity
is available.

──────────────────────────────────────────────────────────────────────────────

CORE CAPABILITIES

Dynamic Tag List Support
Upload an Excel file containing any metadata columns alongside a required
SCADA TAG column. All additional columns are preserved and used as structured
header levels in the exported dataset.

Two Export Modes (No Re-Query Required)

Raw Data Mode
Exports values exactly as returned from PI.

Clean Data Mode
Applies structured cleaning logic:
• Status strings converted to 0 / 1
• Non-numeric values converted to 0
• Negative values clipped to 0
• Known bad tokens handled automatically

Controlled Query Execution
Changing time range, interval, or timezone does NOT automatically query PI.
Data retrieval occurs only when the user clicks:
Run / Refresh PI Query

Professional Excel Output
• Multi-level metadata headers
• Frozen panes
• Numeric formatting to 3 decimals
• Clean structure for downstream analysis

Diagnostics & Transparency
• Tags that fail retrieval are reported separately
• Error messages are displayed clearly
• Time-series plots highlight raw negatives and missing/bad tokens

──────────────────────────────────────────────────────────────────────────────

INPUT FILE STRUCTURE

The uploaded Excel file must contain one column identifying the PI tag.

Accepted column names include:
• SCADA TAG (recommended)
• SCADA_TAG
• TAG
• PI TAG

All additional columns are treated as metadata and are preserved in the output.

Example:

Facility | Unit   | SCADA TAG
---------------------------------------
Plant A  | Pump 1 | TAG.NAME.001
Plant A  | Pump 2 | TAG.NAME.002

──────────────────────────────────────────────────────────────────────────────

TECHNICAL STACK

Frontend      : Streamlit  
Data Layer    : PIconnect (PI SDK access)  
Visualization : Altair  
Export Engine : Pandas + XlsxWriter  
Timezone      : pytz  

Designed for internal deployment environments with PI connectivity.

──────────────────────────────────────────────────────────────────────────────

LOCAL DEPLOYMENT

1) Create virtual environment
2) Install dependencies
3) Launch Streamlit

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Application will be available at:
http://localhost:8501

──────────────────────────────────────────────────────────────────────────────

PRODUCTION DEPLOYMENT OPTIONS

Recommended internal deployment strategies:

• Windows Server with Streamlit service
• Docker-based internal container
• Reverse proxy via IIS or Nginx
• Active Directory authentication integration

Note:
Public cloud deployment is generally not suitable unless PI access is
exposed through a secure internal API layer.

──────────────────────────────────────────────────────────────────────────────

SECURITY NOTES

• Do not embed credentials directly in source code.
• Use environment variables or .streamlit/secrets.toml.
• Bind Streamlit to localhost and expose through a reverse proxy.
• Restrict access using Active Directory groups when possible.

──────────────────────────────────────────────────────────────────────────────

VERSION

Current version: 1.0.0

──────────────────────────────────────────────────────────────────────────────

LICENSE

MIT License  
© 2026 Ali Jozaghi
