from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from scipy.signal import find_peaks
import streamlit as st
import ollama
from pypdf import PdfReader
st.write("NEW VERSION LOADED")
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BioProc AI Assistant",
    page_icon="🧬",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

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

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Generating AI response..."
            ):

                try:

                    prompt = f"""
You are a senior downstream bioprocessing scientist.

Rules:
- Answer scientifically
- Be concise
- Use bullet points
- Avoid greetings
- Focus on practical DSP solutions

Topics:
- Protein A chromatography
- UF/DF
- Aggregation reduction
- HCP reduction
- Resin selection
- CQA and CPP optimization
- DSP troubleshooting

Question:
{user_question}
"""

                    response = ollama.chat(
                        model="phi3:mini",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response["message"]["content"]

                    st.success(
                        "AI Response Generated"
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(f"Error: {e}")

# =====================================================
# PDF SOP ANALYZER
# =====================================================

elif page == "PDF SOP Analyzer":

    st.header("📄 DSP SOP RAG Assistant")

    uploaded_file = st.file_uploader(
        "Upload SOP PDF",
        type="pdf"
    )

    if uploaded_file is not None:

        with open(
            "temp.pdf",
            "wb"
        ) as f:

            f.write(uploaded_file.read())

        try:

            with st.spinner(
                "Processing PDF..."
            ):

                # ---------------------------------
                # LOAD PDF
                # ---------------------------------

                loader = PyPDFLoader(
                    "temp.pdf"
                )

                documents = loader.load()

                # ---------------------------------
                # SPLIT TEXT
                # ---------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                docs = text_splitter.split_documents(
                    documents
                )

                # ---------------------------------
                # EMBEDDINGS
                # ---------------------------------

                embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # ---------------------------------
                # VECTOR DATABASE
                # ---------------------------------

                db = Chroma.from_documents(
                    docs,
                    embedding_model
                )

                st.success(
                    "PDF Processed Successfully"
                )

            # -------------------------------------
            # QUESTION INPUT
            # -------------------------------------

            user_question = st.text_area(
                "Ask question about SOP",
                height=120
            )

            if st.button(
                "Analyze SOP"
            ):

                with st.spinner(
                    "Searching SOP..."
                ):

                    # -----------------------------
                    # RETRIEVE RELEVANT CHUNKS
                    # -----------------------------

                    results = db.similarity_search(
                        user_question,
                        k=3
                    )

                    context = ""

                    for result in results:

                        context += (
                            result.page_content
                            + "\n\n"
                        )

                    # -----------------------------
                    # BUILD PROMPT
                    # -----------------------------

                    prompt = f"""
You are a monoclonal antibody downstream processing expert.

Answer ONLY using the SOP context below.

If answer is not found in context,
say:
'Information not found in SOP.'

SOP CONTEXT:
{context}

QUESTION:
{user_question}

Rules:
- Be concise
- Use DSP terminology
- Avoid hallucination
"""

                    # -----------------------------
                    # OLLAMA RESPONSE
                    # -----------------------------

                    response = ollama.chat(
                        model="phi3:mini",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response["message"]["content"]

                    st.success(
                        "Analysis Complete"
                    )

                    st.markdown(
                        "## 🔬 SOP Answer"
                    )

                    st.write(answer)

                    # -----------------------------
                    # SHOW RETRIEVED CHUNKS
                    # -----------------------------

                    with st.expander(
                        "View Retrieved SOP Chunks"
                    ):

                        st.write(context)

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

    additional_info = st.text_area(
        "Additional Process Details",
        height=150,
        placeholder="""
Example:
- Elution pH = 3.5
- Conductivity = 12 mS/cm
- Flux = 40 LMH
- Hold time = 6 hours
"""
    )

    if st.button("Analyze Problem"):

        with st.spinner(
            "Analyzing process deviation..."
        ):

            try:

                prompt = f"""
You are a senior monoclonal antibody downstream processing scientist
with expertise in:
- Protein A chromatography
- CEX
- AEX
- UF/DF
- Viral filtration
- Aggregation control
- HCP reduction
- DSP scale-up
- mAb purification

Analyze the following downstream bioprocess issue.

ISSUE:
{issue}

PROCESS STEP:
{process_step}

PROCESS DETAILS:
{additional_info}

VERY IMPORTANT RULES:
- Give ONLY biotech downstream processing answers
- Do NOT give generic chemical engineering answers
- Focus on monoclonal antibody purification
- Mention realistic DSP causes
- Mention realistic CPPs and CQAs
- Be highly practical
- Avoid vague explanations

For each issue provide:

1. Most likely DSP-specific root causes
2. CPPs affecting the issue
3. Expected impact on CQAs
4. Immediate corrective actions
5. Optimization experiments
6. Risk severity

Use this scientific style:

ROOT CAUSES:
- cause 1
- cause 2

CPPs:
- pH
- conductivity
- residence time

CQAs IMPACTED:
- aggregates
- HCP
- purity

CORRECTIVE ACTIONS:
- action 1
- action 2

EXPERIMENTS:
- experiment 1
- experiment 2

RISK:
- Low / Moderate / High

Keep answer concise and industry-specific.
"""

                response = ollama.chat(
                    model="phi3:mini",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response["message"]["content"]

                st.success(
                    "Troubleshooting Complete"
                )

                st.write(answer)

            except Exception as e:

                st.error(f"Error: {e}")

# =====================================================
# DSP CALCULATORS
# =====================================================

elif page == "DSP Calculator":

    st.header("📊 DSP Engineering Calculator")

    calculator = st.selectbox(
        "Select Calculator",
        [
            "TMP Calculator",
            "Flux Calculator",
            "Membrane Area Calculator",
            "Pool Dilution Calculator",
            "Resin Sizing Calculator",
            "Load Density Calculator"
        ]
    )

    # =================================================
    # TMP
    # =================================================

    if calculator == "TMP Calculator":

        st.subheader("TMP Calculator")

        feed_pressure = st.number_input(
            "Feed Pressure (psi)",
            value=20.0
        )

        retentate_pressure = st.number_input(
            "Retentate Pressure (psi)",
            value=10.0
        )

        permeate_pressure = st.number_input(
            "Permeate Pressure (psi)",
            value=2.0
        )

        if st.button("Calculate TMP"):

            tmp = (
                (
                    feed_pressure
                    + retentate_pressure
                ) / 2
            ) - permeate_pressure

            st.success(
                f"TMP = {tmp:.2f} psi"
            )

    # =================================================
    # FLUX
    # =================================================

    elif calculator == "Flux Calculator":

        st.subheader("Flux Calculator")

        permeate_flow = st.number_input(
            "Permeate Flow Rate (L/hr)",
            value=50.0
        )

        membrane_area = st.number_input(
            "Membrane Area (m²)",
            value=0.5
        )

        if st.button("Calculate Flux"):

            flux = permeate_flow / membrane_area

            st.success(
                f"Flux = {flux:.2f} LMH"
            )

    # =================================================
    # MEMBRANE AREA
    # =================================================

    elif calculator == "Membrane Area Calculator":

        st.subheader("Membrane Area Calculator")

        volume = st.number_input(
            "Process Volume (L)",
            value=100.0
        )

        flux = st.number_input(
            "Target Flux (LMH)",
            value=50.0
        )

        time = st.number_input(
            "Process Time (hr)",
            value=4.0
        )

        if st.button(
            "Calculate Membrane Area"
        ):

            area = volume / (flux * time)

            st.success(
                f"Required Area = {area:.2f} m²"
            )

    # =================================================
    # DILUTION
    # =================================================

    elif calculator == "Pool Dilution Calculator":

        st.subheader(
            "Pool Dilution Calculator"
        )

        c1 = st.number_input(
            "Initial Conductivity",
            value=20.0
        )

        v1 = st.number_input(
            "Initial Volume (L)",
            value=100.0
        )

        c2 = st.number_input(
            "Target Conductivity",
            value=5.0
        )

        if st.button("Calculate Dilution"):

            final_volume = (c1 * v1) / c2

            buffer_needed = (
                final_volume - v1
            )

            st.success(
                f"Add {buffer_needed:.2f} L buffer"
            )

    # =================================================
    # RESIN
    # =================================================

    elif calculator == "Resin Sizing Calculator":

        st.subheader(
            "Resin Sizing Calculator"
        )

        protein_mass = st.number_input(
            "Protein Mass (g)",
            value=500.0
        )

        dbc = st.number_input(
            "DBC (g/L)",
            value=40.0
        )

        if st.button(
            "Calculate Resin Volume"
        ):

            resin_volume = protein_mass / dbc

            st.success(
                f"Resin Volume = {resin_volume:.2f} L"
            )

    # =================================================
    # LOAD DENSITY
    # =================================================

    elif calculator == "Load Density Calculator":

        st.subheader(
            "Load Density Calculator"
        )

        protein_mass = st.number_input(
            "Protein Mass (mg)",
            value=50000.0
        )

        resin_volume = st.number_input(
            "Resin Volume (mL)",
            value=1000.0
        )

        if st.button(
            "Calculate Load Density"
        ):

            load_density = (
                protein_mass
                / resin_volume
            )

            st.success(
                f"Load Density = {load_density:.2f} mg/mL"
            )
# =====================================================
# CHROMATOGRAM ANALYSIS
# =====================================================

elif page == "Chromatogram Analysis":

    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    st.header("📈 AKTA Chromatogram Analysis")

    uploaded_file = st.file_uploader(
        "Upload Chromatogram CSV",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # =========================================
            # LOAD FILE
            # =========================================

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

            st.success(
                "Chromatogram Loaded Successfully"
            )

            st.subheader("Preview Data")

            st.dataframe(df.head())

            # =========================================
            # COLUMN SELECTION
            # =========================================

            x_column = st.selectbox(
                "Select X Axis",
                df.columns
            )

            y_column = st.selectbox(
                "Select Y Axis",
                df.columns
            )

            # =========================================
            # ANALYSIS BUTTON
            # =========================================

            if st.button("Analyze Chromatogram"):

                # -------------------------------------
                # PLOT
                # -------------------------------------

                fig, ax = plt.subplots(
                    figsize=(10, 4)
                )

                ax.plot(
                    df[x_column],
                    df[y_column]
                )

                ax.set_xlabel(x_column)

                ax.set_ylabel(y_column)

                ax.set_title(
                    "Chromatogram"
                )

                st.pyplot(fig)

                # -------------------------------------
                # PEAK DETECTION
                # -------------------------------------

                peaks, properties = find_peaks(
                    df[y_column],
                    height=10,
                    distance=1
                )

                peak_values = (
                    df[y_column]
                    .iloc[peaks]
                    .values
                )

                peak_positions = (
                    df[x_column]
                    .iloc[peaks]
                    .values
                )

                st.subheader(
                    "Detected Peaks"
                )

                if len(peaks) == 0:

                    st.warning(
                        "No significant peaks detected"
                    )

                else:

                    for i in range(len(peaks)):

                        st.write(
                            f"Peak {i+1}"
                        )

                        st.write(
                            f"Position: {peak_positions[i]}"
                        )

                        st.write(
                            f"Height: {peak_values[i]}"
                        )

                        st.write("---")

                # -------------------------------------
                # SPIKE DETECTION
                # -------------------------------------

                mean_signal = (
                    df[y_column].mean()
                )

                threshold = (
                    mean_signal * 3
                )

                spikes = df[
                    df[y_column] > threshold
                ]

                st.subheader(
                    "Signal Spike Detection"
                )

                st.write(
                    f"Spike Points Detected: {len(spikes)}"
                )

                # -------------------------------------
                # AI INTERPRETATION
                # -------------------------------------

                summary = f"""
Chromatogram Analysis Summary

Detected Peaks:
{len(peaks)}

Peak Positions:
{peak_positions}

Peak Heights:
{peak_values}

Spike Count:
{len(spikes)}

Provide downstream processing interpretation.

Discuss:
- peak symmetry
- possible aggregation
- overloading
- pooling concerns
- purification quality
- process optimization
"""

                response = ollama.chat(
                    model="phi3:mini",
                    messages=[
                        {
                            "role": "user",
                            "content": summary
                        }
                    ]
                )

                answer = (
                    response["message"]["content"]
                )

                st.subheader(
                    "AI DSP Interpretation"
                )

                st.markdown(answer)

        except Exception as e:

            st.error(f"Error: {e}")
            # =====================================================
# CHROMATOGRAM COMPARISON
# =====================================================

elif page == "Chromatogram Comparison":

    import pandas as pd
    import matplotlib.pyplot as plt

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

            st.success(
                "Both chromatograms loaded"
            )

            x1 = st.selectbox(
                "Run 1 X Axis",
                df1.columns,
                key="x1"
            )

            y1 = st.selectbox(
                "Run 1 Y Axis",
                df1.columns,
                key="y1"
            )

            x2 = st.selectbox(
                "Run 2 X Axis",
                df2.columns,
                key="x2"
            )

            y2 = st.selectbox(
                "Run 2 Y Axis",
                df2.columns,
                key="y2"
            )

            if st.button(
                "Compare Runs"
            ):

                # ==================================
                # OVERLAY PLOT
                # ==================================

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                ax.plot(
                    df1[x1],
                    df1[y1],
                    label="Run 1"
                )

                ax.plot(
                    df2[x2],
                    df2[y2],
                    label="Run 2"
                )

                ax.set_xlabel("Volume")

                ax.set_ylabel("UV")

                ax.set_title(
                    "Chromatogram Overlay"
                )

                ax.legend()

                st.pyplot(fig)

                # ==================================
                # BASIC METRICS
                # ==================================

                peak1 = df1[y1].max()

                peak2 = df2[y2].max()

                shift = peak2 - peak1

                st.subheader(
                    "Run Comparison Metrics"
                )

                st.write(
                    f"Run 1 Peak: {peak1}"
                )

                st.write(
                    f"Run 2 Peak: {peak2}"
                )

                st.write(
                    f"Peak Difference: {shift}"
                )

                # ==================================
                # AI ANALYSIS
                # ==================================

                comparison_prompt = f"""
Compare two downstream chromatography runs.

RUN 1 PEAK:
{peak1}

RUN 2 PEAK:
{peak2}

PEAK DIFFERENCE:
{shift}

Provide:
- process drift analysis
- possible resin aging
- overloading indication
- pooling concerns
- process consistency interpretation
- optimization recommendations
"""

                response = ollama.chat(
                    model="phi3:mini",
                    messages=[
                        {
                            "role": "user",
                            "content": comparison_prompt
                        }
                    ]
                )

                answer = (
                    response["message"]["content"]
                )

                st.subheader(
                    "AI Comparison Interpretation"
                )

                st.markdown(answer)

        except Exception as e:

            st.error(f"Error: {e}")
            # =====================================================
# POOLING RECOMMENDATION
# =====================================================

elif page == "Pooling Recommendation":

    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    st.header("🧪 AI Pooling Recommendation")

    uploaded_file = st.file_uploader(
        "Upload Chromatogram",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # =====================================
            # LOAD FILE
            # =====================================

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

            st.success(
                "Chromatogram Loaded"
            )

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

                # =================================
                # PLOT
                # =================================

                fig, ax = plt.subplots(
                    figsize=(10, 4)
                )

                ax.plot(
                    df[x_column],
                    df[y_column]
                )

                ax.set_title(
                    "Chromatogram"
                )

                st.pyplot(fig)

                # =================================
                # PEAK DETECTION
                # =================================

                peaks, properties = find_peaks(
                    df[y_column],
                    height=10
                )

                if len(peaks) == 0:

                    st.warning(
                        "No major peaks detected"
                    )

                else:

                    # -----------------------------
                    # MAIN PEAK
                    # -----------------------------

                    highest_peak_index = peaks[
                        properties["peak_heights"].argmax()
                    ]

                    peak_position = df.loc[
                        highest_peak_index,
                        x_column
                    ]

                    peak_height = df.loc[
                        highest_peak_index,
                        y_column
                    ]

                    # -----------------------------
                    # POOL WINDOW
                    # -----------------------------

                    peak_index = int(
                        df[y_column].idxmax()
                    )

                    left = max(
                        peak_index - 1,
                        0
                    )

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
                    # -----------------------------
                    # AI ANALYSIS
                    # -----------------------------

                    prompt = f"""
You are a monoclonal antibody DSP scientist.

Analyze this chromatography peak.

Peak Position:
{peak_position}

Peak Height:
{peak_height}

Pooling Window:
{pooling_start} to {pooling_end}

Provide:
- pooling suitability
- impurity risk
- aggregate risk
- shoulder peak concern
- pooling optimization strategy
- process recommendations

Use concise DSP terminology.
"""

                    response = ollama.chat(
                        model="phi3:mini",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = (
                        response["message"]["content"]
                    )

                    st.subheader(
                        "AI Pooling Assessment"
                    )

                    st.markdown(answer)

        except Exception as e:

            st.error(f"Error: {e}")