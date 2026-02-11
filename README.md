\# PI Data Extractor (Streamlit)



A Streamlit app that:

\- accepts an Excel “tag list” (any metadata columns + a required \*\*SCADA TAG\*\* column),

\- queries PI for interpolated time-series data,

\- exports \*\*Raw\*\* or \*\*Clean\*\* data to a formatted Excel file,

\- visualizes any selected tag with quality markers (raw negatives, missing/bad tokens).



\## Features

\- \*\*Dynamic metadata headers:\*\* any extra columns in the uploaded tag list become Excel header rows

\- \*\*Two export modes (no re-query):\*\*

&nbsp; - \*\*Raw data:\*\* as returned by PI

&nbsp; - \*\*Clean data:\*\* status strings → 0/1; non-numeric → 0; negatives → 0

\- \*\*No auto-run:\*\* changing inputs does \*not\* query PI until you click \*\*Run / Refresh PI Query\*\*

\- \*\*Download-ready Excel:\*\* numeric format to 3 decimals, frozen headers

\- \*\*Diagnostics:\*\* shows tags that failed and error messages



---



\## Input file format

Your Excel must contain a tag column named one of:

\- `SCADA TAG` (recommended)

\- `SCADA\_TAG`, `TAG`, `PI TAG`, etc.



All other columns are treated as metadata and can be blank.



Example:



| Facility | Unit | SCADA TAG |

|---|---|---|

| Plant A | Pump 1 | TAG.NAME.001 |

| Plant A | Pump 2 | TAG.NAME.002 |



---



\## Local installation (recommended for PI environments)

> PI connectivity usually requires internal network access and/or domain authentication.



\### 1) Create and activate a virtual environment

```bash

python -m venv .venv

\# Windows:

.venv\\Scripts\\activate

\# macOS/Linux:

source .venv/bin/activate



