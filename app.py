# app.py
import io
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt
import pytz

from PIconnect.PI import PIServer


# -----------------------------
# Constants / Mappings
# -----------------------------
TAG_COL = "SCADA TAG"

# Only for recognizing the tag column (everything else is treated as generic metadata)
TAG_ALIASES = {
    "SCADA TAG": TAG_COL,
    "SCADA_TAG": TAG_COL,
    "SCADATAG": TAG_COL,
    "TAG": TAG_COL,
    "PI TAG": TAG_COL,
    "PITAG": TAG_COL,
}

# Strings that mean "bad/missing" (case-insensitive, after strip)
BAD_TOKENS = {
    "NO DATA",
    "NODATA",
    "I/O TIMEOUT",
    "IO TIMEOUT",
    "TIMEOUT",
    "BAD INPUT",
    "BAD",
    "ERROR",
    "DISCONNECTED",
    "COMM FAIL",
    "COMMUNICATION FAIL",
    "NOT CONNECTED",
    "NULL",
    "NAN",
    "",
}

# Status mapping (case-insensitive)
STATUS_MAP = {
    "ACTIVE": 1,
    "INACTIVE": 0,
    "ON": 1,
    "OFF": 0,
    "RUN": 1,
    "RUNNING": 1,
    "STOP": 0,
    "STOPPED": 0,
    "START": 1,
    "STARTED": 1,
    "OPEN": 1,
    "OPENED": 1,
    "CLOSE": 0,
    "CLOSED": 0,
    "SHUT": 0,
    "SHUTTED": 0,
    "TRUE": 1,
    "FALSE": 0,
    "YES": 1,
    "NO": 0,
}


# -----------------------------
# Normalization / Validation
# -----------------------------
def _normalize_colname(c: str) -> str:
    """
    Normalize any column name:
      - strip
      - uppercase
      - map common tag aliases to TAG_COL
    """
    raw = str(c).strip()
    up = raw.upper()

    compact = up.replace(" ", "").replace("-", "_")
    # Try aliasing both forms
    if up in TAG_ALIASES:
        return TAG_ALIASES[up]
    if compact in TAG_ALIASES:
        return TAG_ALIASES[compact]
    return up


def _validate_taglist_df(df: pd.DataFrame):
    """
    General taglist validation:
      - Normalizes column names to ALL CAPS
      - Requires a SCADA TAG column (via aliases)
      - Treats ALL other columns as metadata header rows (dynamic!)
      - Drops rows with blank SCADA TAG
    Returns:
      tag_df (cleaned)
      out_cols (metadata cols in input order + SCADA TAG at end)
    """
    df = df.copy()
    original_cols = list(df.columns)
    df.columns = [_normalize_colname(c) for c in df.columns]

    if TAG_COL not in df.columns:
        raise ValueError(
            f"Input Excel must contain a tag column like 'SCADA Tag' / 'SCADA_TAG' / 'TAG'. "
            f"Found columns: {list(df.columns)}"
        )

    # Preserve ordering based on original file order
    norm_map = {orig: _normalize_colname(orig) for orig in original_cols}
    ordered_norm = [norm_map[orig] for orig in original_cols]

    meta_cols = [c for c in ordered_norm if c != TAG_COL]
    out_cols = meta_cols + [TAG_COL]  # always include SCADA TAG last

    # Drop duplicate columns safely
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df[[c for c in out_cols if c in df.columns]].copy()

    # Clean tag values
    df[TAG_COL] = df[TAG_COL].astype(str).str.strip()
    df = df[df[TAG_COL].ne("") & df[TAG_COL].notna()].reset_index(drop=True)

    return df, out_cols


# -----------------------------
# Cleaning / Parsing
# -----------------------------
def _is_bad_token(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().upper()
    return s in BAD_TOKENS


def _status_to_numeric(x):
    """
    Convert common status strings to 0/1. Return None if not a status token.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()

    if s in STATUS_MAP:
        return STATUS_MAP[s]

    # Handle combined strings like "ACTIVE/INACTIVE", "ON - OFF", etc.
    if any(k in s for k in ["ACTIVE", "RUN", "RUNNING", "ON", "OPEN", "TRUE", "YES", "START"]):
        return 1
    if any(k in s for k in ["INACTIVE", "STOP", "STOPPED", "OFF", "CLOSE", "CLOSED", "FALSE", "NO", "SHUT"]):
        return 0

    return None


def to_numeric_with_status(series: pd.Series) -> pd.Series:
    """
    Convert a series to numeric, but also map status strings -> 0/1.
    Anything else non-numeric becomes NaN.
    """
    mapped = series.map(_status_to_numeric)
    numeric = pd.to_numeric(series, errors="coerce")

    out = numeric.copy()
    mask_status = pd.Series(mapped).notna().values
    out.iloc[mask_status] = pd.Series(mapped).iloc[mask_status].astype(float).values
    return out


def clean_pi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean PI data:
      - status strings -> 0/1
      - other non-numeric -> NaN
      - NaN -> 0
      - negatives -> 0
    """
    cleaned = df.copy()
    for c in cleaned.columns:
        cleaned[c] = to_numeric_with_status(cleaned[c])
    cleaned = cleaned.fillna(0)
    cleaned = cleaned.clip(lower=0)
    return cleaned


# -----------------------------
# PI Query helpers
# -----------------------------
def _pi_interpolated_series(pt, start_iso: str, end_iso: str, interval: str) -> pd.Series:
    s_interp = pt.interpolated_values(start_iso, end_iso, interval)
    if isinstance(s_interp, pd.DataFrame):
        s_out = s_interp.iloc[:, 0]
    else:
        s_out = s_interp
    if not isinstance(s_out, pd.Series):
        s_out = pd.Series(s_out)
    return s_out


def _utc_to_local_excel_naive(dt_index, tz: str) -> pd.DatetimeIndex:
    idx = pd.to_datetime(dt_index, utc=True)
    idx = idx.tz_convert(tz)
    idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx)


def build_output_dataframe(
    taglist: pd.DataFrame,
    out_cols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    interval: str,
    tz: str,
):
    start_iso = pd.Timestamp(start_dt, tz=tz).isoformat()
    end_iso = pd.Timestamp(end_dt, tz=tz).isoformat()

    series_list = []
    col_tuples = []
    error_rows = []

    with PIServer() as pi:
        for _, row in taglist.iterrows():
            tag = str(row[TAG_COL])
            try:
                pt = pi.search(tag)[0]
                s = _pi_interpolated_series(pt, start_iso, end_iso, interval)
                s.index = _utc_to_local_excel_naive(s.index, tz)

                series_list.append(s)
                col_tuples.append(tuple(str(row[c]) if c in row else "" for c in out_cols))

            except Exception as e:
                err = {c: str(row[c]) if c in row else "" for c in out_cols}
                err["ERROR"] = str(e)
                error_rows.append(err)

    if not series_list:
        raise RuntimeError("No PI data was retrieved. Check tags/permissions/date range.")

    out = pd.concat(series_list, axis=1)
    out.columns = pd.MultiIndex.from_tuples(col_tuples)

    errors_df = pd.DataFrame(error_rows) if error_rows else pd.DataFrame(columns=out_cols + ["ERROR"])
    return out, errors_df


# -----------------------------
# Excel writer (3 decimals)
# -----------------------------
def dataframe_to_excel_bytes(data_df: pd.DataFrame, header_rows: list[str]) -> bytes:
    """
    Writes ONLY one sheet 'Data':
      - Column A rows 1..N contain header row labels (column names from input, ALL CAPS)
      - Header values across columns are MultiIndex level values
      - Data begins after header rows
    Formats numeric cells to 3 decimals.
    """
    if not isinstance(data_df.columns, pd.MultiIndex):
        raise ValueError("data_df.columns must be a MultiIndex")

    nlevels = data_df.columns.nlevels
    if nlevels != len(header_rows):
        raise ValueError(f"Header rows ({len(header_rows)}) must match MultiIndex levels ({nlevels}).")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        sheet_name = "Data"
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        num_fmt_3 = workbook.add_format({"num_format": "0.000"})  # 3 decimals

        # Header row labels in column A (A1..A{nlevels})
        for r, label in enumerate(header_rows):
            worksheet.write(r, 0, label.upper())

        # Header values across columns
        col_levels = [data_df.columns.get_level_values(i) for i in range(nlevels)]
        ncols = data_df.shape[1]
        for j in range(ncols):
            for r in range(nlevels):
                worksheet.write(r, j + 1, str(col_levels[r][j]))

        # Data after header rows
        flat_df = data_df.copy()
        flat_df.columns = [""] * ncols
        flat_df.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=nlevels,
            startcol=0,
            header=False,
            index=True,
        )

        worksheet.freeze_panes(nlevels, 1)
        worksheet.set_column(0, 0, 22)  # datetime column width
        worksheet.set_column(1, ncols, 14, num_fmt_3)  # values to 3 decimals

    return output.getvalue()


# -----------------------------
# Plot helpers
# -----------------------------
def _col_label(col_tuple) -> str:
    parts = [str(x).strip() for x in col_tuple]
    parts = [p for p in parts if p and p.lower() != "nan"]
    return " | ".join(parts) if parts else str(col_tuple)


def _build_fancy_chart(
    export_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    col,
    label: str,
    show_points: bool = True,
    show_negative_markers: bool = True,
    show_missing_markers: bool = True,
):
    """
    Plot exported data (same as Excel) as numeric (status->0/1 supported).
    Highlights based on RAW:
      - raw_negative: only when raw numeric exists and < 0
      - raw_missing_bad: true missing OR known BAD tokens
    Show table: timestamp, raw_value, export_value, raw_negative, raw_missing_bad
    """
    raw_series = raw_df[col]
    raw_value = raw_series.map(lambda x: None if pd.isna(x) else str(x))

    raw_missing_bad = raw_series.map(_is_bad_token)

    raw_numeric = pd.to_numeric(raw_series, errors="coerce")
    raw_negative = (raw_numeric < 0) & (~raw_numeric.isna())

    export_numeric = to_numeric_with_status(export_df[col])

    plot_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(export_df.index, errors="coerce"),
            "raw_value": raw_value.values,
            "export_value": export_numeric.values,
            "raw_negative": raw_negative.values,
            "raw_missing_bad": raw_missing_bad.values,
        }
    )

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Raw negatives", int(plot_df["raw_negative"].sum()))
    with m2:
        st.metric("Raw missing/bad", int(plot_df["raw_missing_bad"].sum()))
    with m3:
        st.metric("Export numeric points", int(pd.Series(plot_df["export_value"]).notna().sum()))

    hover = alt.selection_point(fields=["timestamp"], nearest=True, on="mouseover", empty=False, clear="mouseout")

    base = alt.Chart(plot_df).encode(
        x=alt.X("timestamp:T", title="Time"),
    )

    line = base.mark_line().encode(
        y=alt.Y("export_value:Q", title="Value"),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("raw_value:N", title="Raw value"),
            alt.Tooltip("export_value:Q", title="Export value"),
            alt.Tooltip("raw_negative:N", title="Raw negative?"),
            alt.Tooltip("raw_missing_bad:N", title="Raw missing/bad?"),
        ],
    )

    layers = [line]

    if show_points:
        layers.append(base.mark_point(filled=True, size=28, opacity=0.55).encode(y="export_value:Q"))

    layers.append(base.mark_rule(opacity=0.25).add_params(hover).transform_filter(hover))
    layers.append(
        base.mark_point(size=80, filled=True).encode(y="export_value:Q").add_params(hover).transform_filter(hover)
    )

    if show_negative_markers and int(plot_df["raw_negative"].sum()) > 0:
        layers.append(
            base.transform_filter("datum.raw_negative == true")
            .mark_point(filled=True, size=90)
            .encode(y="export_value:Q", color=alt.value("#d62728"))
        )

    if show_missing_markers and int(plot_df["raw_missing_bad"].sum()) > 0:
        layers.append(
            base.transform_filter("datum.raw_missing_bad == true")
            .mark_point(filled=True, size=70, shape="triangle-up")
            .encode(y=alt.value(0), color=alt.value("#ff7f0e"))
        )

    chart = alt.layer(*layers).properties(height=420, title=label).interactive()
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Show plotted data", expanded=False):
        st.dataframe(plot_df, use_container_width=True)


# -----------------------------
# Signature helpers (for UI message only)
# -----------------------------
def _file_hash(uploaded_file) -> str:
    b = uploaded_file.getvalue()
    return hashlib.md5(b).hexdigest()


def _query_signature(file_md5: str, start_full: datetime, end_full: datetime, interval: str, tz: str) -> str:
    s = f"{file_md5}|{start_full.isoformat()}|{end_full.isoformat()}|{interval}|{tz}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _build_cached_exports():
    """Build both raw + clean excel bytes from the cached raw_df (fast, no PI call)."""
    raw_df = st.session_state["raw_df"]
    out_cols = st.session_state["out_cols"]
    if raw_df is None or out_cols is None:
        return

    st.session_state["raw_xbytes"] = dataframe_to_excel_bytes(raw_df, out_cols)
    st.session_state["clean_xbytes"] = dataframe_to_excel_bytes(clean_pi_dataframe(raw_df), out_cols)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PI Data Extractor", layout="wide")

st.title("📊 PI Data Extractor")
st.caption("Upload any tag list (any metadata columns) → pull PI time series → export to Excel + plot.")

uploaded = st.file_uploader("Upload tag list Excel", type=["xlsx", "xls"])

colA, colB, colC, colD = st.columns(4)
with colA:
    start_dt = st.date_input("Start date", value=datetime(2025, 8, 2).date())
    start_time = st.time_input("Start time", value=datetime(2025, 8, 2, 0, 0).time())
with colB:
    end_dt = st.date_input("End date", value=datetime(2025, 8, 9).date())
    end_time = st.time_input("End time", value=datetime(2025, 8, 9, 0, 0).time())
with colC:
    interval_choice = st.selectbox("Interval", options=["1m", "1h", "1d", "Custom..."], index=1)
    interval_custom = st.text_input(
        "Custom interval", value="15m", disabled=(interval_choice != "Custom..."), help="Examples: 15m, 5m, 2h"
    )
    interval = interval_choice if interval_choice != "Custom..." else interval_custom.strip()
with colD:
    tz_list = pytz.common_timezones
    default_tz = "America/Chicago"
    default_idx = tz_list.index(default_tz) if default_tz in tz_list else 0
    tz_choice = st.selectbox("Timezone", options=tz_list + ["Custom..."], index=default_idx)
    tz_custom = st.text_input("Custom timezone", value=default_tz, disabled=(tz_choice != "Custom..."))
    tz = tz_choice if tz_choice != "Custom..." else tz_custom.strip()

data_mode = st.radio(
    "Data to export",
    options=["Raw data", "Clean data"],
    index=1,
    help="Raw = PI values as returned. Clean = status strings→0/1, non-numeric→0, negatives→0.",
)

run_btn = st.button("Run / Refresh PI Query", type="primary")

# Session state
for key in [
    "sig",
    "raw_df",
    "errors_df",
    "out_cols",
    "raw_xbytes",
    "clean_xbytes",
    "last_start_full",
    "last_end_full",
    "last_interval",
    "last_tz",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if uploaded is None:
    st.info("Upload your tag list Excel to begin.")
else:
    start_full = datetime.combine(start_dt, start_time)
    end_full = datetime.combine(end_dt, end_time)

    if end_full <= start_full:
        st.error("End datetime must be after start datetime.")
    else:
        # Compute signature for "inputs changed" message (NO auto-run!)
        file_md5 = _file_hash(uploaded)
        sig = _query_signature(file_md5, start_full, end_full, interval, tz)

        # Message if inputs changed but user didn't click Run
        if st.session_state["sig"] is not None and st.session_state["sig"] != sig and not run_btn:
            st.info("Inputs changed. Click **Run / Refresh PI Query** to fetch new data.")

        if run_btn:
            try:
                tag_df_raw = pd.read_excel(uploaded)
                tag_df, out_cols = _validate_taglist_df(tag_df_raw)

                with st.spinner("Querying PI (raw) and building export cache..."):
                    raw_df, errors_df = build_output_dataframe(tag_df, out_cols, start_full, end_full, interval, tz)

                    st.session_state["raw_df"] = raw_df
                    st.session_state["errors_df"] = errors_df
                    st.session_state["out_cols"] = out_cols
                    st.session_state["sig"] = sig

                    st.session_state["last_start_full"] = start_full
                    st.session_state["last_end_full"] = end_full
                    st.session_state["last_interval"] = interval
                    st.session_state["last_tz"] = tz

                    _build_cached_exports()

                st.success(f"Export cache ready: {raw_df.shape[0]} rows × {raw_df.shape[1]} tags")

                if errors_df is not None and not errors_df.empty:
                    st.warning(f"{len(errors_df)} tag(s) had errors (shown below, not included in Excel).")
                    st.dataframe(errors_df, use_container_width=True)

            except Exception as e:
                st.error(f"Could not read/validate/query PI: {e}")

# -----------------------------
# Download + Preview + Plots (no PI re-query)
# -----------------------------
if st.session_state["raw_df"] is not None:
    raw_df = st.session_state["raw_df"]
    out_cols = st.session_state["out_cols"]

    # Choose export df based on mode (no PI call)
    export_df = raw_df if data_mode.startswith("Raw") else clean_pi_dataframe(raw_df)

    # Prefer cached Excel bytes for speed
    if data_mode.startswith("Raw"):
        xbytes = st.session_state["raw_xbytes"]
    else:
        xbytes = st.session_state["clean_xbytes"]

    # Filename (based on last successful run)
    s0 = st.session_state["last_start_full"]
    s1 = st.session_state["last_end_full"]
    itv = st.session_state["last_interval"]
    tz_used = st.session_state["last_tz"]
    if s0 and s1 and itv:
        fname = f"PI_export_{data_mode.replace(' ', '_')}_{s0:%Y-%m-%d_%H%M}_to_{s1:%Y-%m-%d_%H%M}_{itv}_{tz_used}.xlsx"
    else:
        fname = f"PI_export_{data_mode.replace(' ', '_')}.xlsx"

    st.subheader("Download")
    st.download_button(
        label=f"⬇️ Download Excel ({data_mode})",
        data=xbytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Data Preview")
    st.dataframe(export_df.head(5), use_container_width=True)

    st.divider()
    st.subheader("Time-Series Plots")

    # Build label map from columns
    label_to_col = {}
    labels = []
    for col in export_df.columns:
        lbl = _col_label(col)
        base = lbl
        k = 2
        while lbl in label_to_col:
            lbl = f"{base} ({k})"
            k += 1
        label_to_col[lbl] = col
        labels.append(lbl)

    selected_label = st.selectbox("Select a tag/column to plot", options=labels)
    col = label_to_col[selected_label]

    c1, c2, c3 = st.columns(3)
    with c1:
        show_points = st.checkbox("Show points", value=True)
    with c2:
        show_negative = st.checkbox("Highlight raw negatives", value=True)
    with c3:
        show_missing = st.checkbox("Highlight raw missing/bad", value=True)

    if data_mode.startswith("Clean"):
        st.info("Plot uses cleaned/exported values. Markers show where RAW was negative or missing/bad.")
    else:
        st.info("Plot uses exported values (raw). Markers show raw negatives and missing/bad.")

    _build_fancy_chart(
        export_df=export_df,
        raw_df=raw_df,
        col=col,
        label=selected_label,
        show_points=show_points,
        show_negative_markers=show_negative,
        show_missing_markers=show_missing,
    )
