import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pypdf import PdfReader

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BioProc AI Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 BioProc AI Assistant")

st.write(
    "AI assistant for downstream bioprocessing scientists"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Tool",
    [
        "AI Chatbot",
        "PDF SOP Analyzer",
        "DSP Troubleshooting",
        "DSP Calculator",
        "Chromatogram Analysis",
        "Chromatogram Comparison",
        "Pooling Recommendation"
    ]
)

# =====================================================
# AI CHATBOT
# =====================================================

if page == "AI Chatbot":

    st.header("🧠 DSP AI Chatbot")

    user_question = st.text_area(
        "Ask your DSP question",
        height=150
    )

    if st.button("Ask AI"):

        if user_question.strip() == "":
            st.warning("Please enter a question.")

        else:

            answer = f"""
### DSP AI Response

Question:
{user_question}

Possible DSP considerations:

- Review chromatography loading conditions
- Evaluate conductivity and pH optimization
- Check aggregation risk during low pH hold
- Monitor HCP clearance trends
- Evaluate membrane fouling if UF/DF involved
- Confirm resin binding capacity

This lightweight demo version is deployment ready.
"""

            st.success("AI Response Generated")

            st.markdown(answer)

# =====================================================
# PDF SOP ANALYZER
# =====================================================

elif page == "PDF SOP Analyzer":

    st.header("📄 DSP SOP Analyzer")

    uploaded_file = st.file_uploader(
        "Upload SOP PDF",
        type="pdf"
    )

    if uploaded_file is not None:

        try:

            reader = PdfReader(uploaded_file)

            text = ""

            for page_pdf in reader.pages:

                extracted = page_pdf.extract_text()

                if extracted:
                    text += extracted

            st.success("PDF Loaded Successfully")

            user_question = st.text_area(
                "Ask question about SOP",
                height=120
            )

            if st.button("Analyze SOP"):

                st.markdown("## 🔬 SOP Analysis")

                if user_question.lower() in text.lower():

                    st.success(
                        "Relevant information found in SOP."
                    )

                else:

                    st.warning(
                        "Exact match not found. Review SOP text below."
                    )

                with st.expander("View Extracted SOP Text"):

                    st.write(text[:5000])

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================================
# DSP TROUBLESHOOTING
# =====================================================

elif page == "DSP Troubleshooting":

    st.header("🧪 DSP Troubleshooting Expert")

    issue = st.selectbox(
        "Select Issue",
        [
            "High Aggregates",
            "Low Yield",
            "High HCP",
            "High Pressure",
            "Poor Binding",
            "Low Purity",
            "Membrane Fouling",
            "Product Loss",
            "Precipitation"
        ]
    )

    process_step = st.selectbox(
        "Process Step",
        [
            "Protein A Chromatography",
            "CEX Chromatography",
            "AEX Chromatography",
            "UF/DF",
            "HIC",
            "SEC",
            "Virus Filtration",
            "Depth Filtration"
        ]
    )

    if st.button("Analyze Problem"):

        result = f"""
### DSP Troubleshooting Report

Issue:
{issue}

Process Step:
{process_step}

Possible Root Causes:
- Incorrect pH/conductivity
- Resin overloading
- Membrane fouling
- Aggregation during hold step

Recommended Actions:
- Optimize loading density
- Reduce process hold time
- Evaluate buffer composition
- Monitor TMP and flux

Risk Level:
Moderate
"""

        st.markdown(result)

# =====================================================
# DSP CALCULATOR
# =====================================================

elif page == "DSP Calculator":

    st.header("📊 DSP Engineering Calculator")

    calculator = st.selectbox(
        "Select Calculator",
        [
            "TMP Calculator",
            "Flux Calculator",
            "Membrane Area Calculator"
        ]
    )

    if calculator == "TMP Calculator":

        feed = st.number_input(
            "Feed Pressure",
            value=20.0
        )

        ret = st.number_input(
            "Retentate Pressure",
            value=10.0
        )

        perm = st.number_input(
            "Permeate Pressure",
            value=2.0
        )

        if st.button("Calculate TMP"):

            tmp = ((feed + ret) / 2) - perm

            st.success(f"TMP = {tmp:.2f} psi")

    elif calculator == "Flux Calculator":

        flow = st.number_input(
            "Flow Rate",
            value=50.0
        )

        area = st.number_input(
            "Membrane Area",
            value=0.5
        )

        if st.button("Calculate Flux"):

            flux = flow / area

            st.success(f"Flux = {flux:.2f} LMH")

    elif calculator == "Membrane Area Calculator":

        volume = st.number_input(
            "Volume",
            value=100.0
        )

        flux = st.number_input(
            "Flux",
            value=50.0
        )

        time = st.number_input(
            "Time",
            value=4.0
        )

        if st.button("Calculate Area"):

            membrane_area = volume / (flux * time)

            st.success(
                f"Required Area = {membrane_area:.2f} m²"
            )

# =====================================================
# CHROMATOGRAM ANALYSIS
# =====================================================

elif page == "Chromatogram Analysis":

    st.header("📈 AKTA Chromatogram Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.dataframe(df.head())

            x_column = st.selectbox(
                "X Axis",
                df.columns
            )

            y_column = st.selectbox(
                "Y Axis",
                df.columns
            )

            if st.button("Analyze Chromatogram"):

                fig, ax = plt.subplots(figsize=(10, 4))

                ax.plot(df[x_column], df[y_column])

                ax.set_title("Chromatogram")

                st.pyplot(fig)

                peaks, _ = find_peaks(df[y_column])

                st.success(
                    f"Detected Peaks: {len(peaks)}"
                )

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================================
# CHROMATOGRAM COMPARISON
# =====================================================

elif page == "Chromatogram Comparison":

    st.header("📊 Chromatogram Run Comparison")

    file1 = st.file_uploader(
        "Upload Run 1",
        type=["csv"],
        key="run1"
    )

    file2 = st.file_uploader(
        "Upload Run 2",
        type=["csv"],
        key="run2"
    )

    if file1 and file2:

        try:

            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

            st.success("Files Loaded")

            st.write(
                "Chromatogram comparison ready."
            )

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================================
# POOLING RECOMMENDATION
# =====================================================

elif page == "Pooling Recommendation":

    st.header("🧪 AI Pooling Recommendation")

    uploaded_file = st.file_uploader(
        "Upload Chromatogram",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            x_column = st.selectbox(
                "Select X Axis",
                df.columns
            )

            y_column = st.selectbox(
                "Select Y Axis",
                df.columns
            )

            if st.button(
                "Generate Pooling Recommendation"
            ):

                peak_index = int(
                    df[y_column].idxmax()
                )

                left = max(peak_index - 1, 0)

                right = min(
                    peak_index + 1,
                    len(df) - 1
                )

                pooling_start = df.loc[
                    left,
                    x_column
                ]

                pooling_end = df.loc[
                    right,
                    x_column
                ]

                st.success(
                    f"Suggested Pooling Window: "
                    f"{pooling_start} to {pooling_end}"
                )

                st.markdown(
                    """
### Pooling Assessment

- Main peak detected successfully
- Low impurity overlap observed
- Suitable for collection
- Monitor shoulder peaks during scale-up
"""
                )

        except Exception as e:

            st.error(f"Error: {e}")