"""
BRAIN-FM // EVALUATION HARNESS
Systematic Benchmarking & Evaluation of Structural MRI Foundation Models
Strict Black & White Minimalist Web Interface.

Run with either:
  python app.py
or:
  streamlit run app.py
"""

import os
import sys

def run_streamlit_app():
    import streamlit as st
    from ui.styles import BLACK_AND_WHITE_CSS
    from ui.views_overview import render_overview_view
    from ui.views_benchmark import render_benchmark_view
    from ui.views_results import render_results_view
    from ui.views_correlation import render_correlation_view
    from ui.views_sensitivity import render_sensitivity_view
    from ui.views_data import render_data_view

    # Configure page
    st.set_page_config(
        page_title="BRAIN-FM // BENCHMARK",
        page_icon="⚪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject strict Black & White styling
    st.markdown(BLACK_AND_WHITE_CSS, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #222222; margin-bottom: 20px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em; color: #888888;">
                EPICS-LAB // RESEARCH
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin-top: 4px;">
                BRAIN-FM
            </div>
            <div style="font-size: 0.8rem; color: #AAAAAA; font-family: 'JetBrains Mono', monospace;">
                EVALUATION HARNESS
            </div>
            <div style="margin-top: 10px;">
                <span class="bw-badge bw-badge-solid">MONOCHROME</span>
                <span class="bw-badge bw-badge-outline" style="margin-left: 4px;">V2.4</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        menu_options = [
            "01 // OVERVIEW & ARCHITECTURE",
            "02 // BENCHMARK SUITE",
            "03 // RESULTS & METRICS",
            "04 // CORRELATION & CLUSTERING",
            "05 // PREPROCESSING SENSITIVITY",
            "06 // DATA & CONFIGURATION"
        ]

        choice = st.radio(
            "NAVIGATION",
            menu_options,
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #666666; line-height: 1.7;">
            <div><strong>MODELS:</strong> 5 SSL + 1 Untrained</div>
            <div><strong>REFERENCE:</strong> FreeSurfer</div>
            <div><strong>COHORTS:</strong> PPMI, HBN, NKI</div>
            <div><strong>MODE:</strong> Offline Zero-ML</div>
            <div><strong>PALETTE:</strong> Strict B&W</div>
        </div>
        """, unsafe_allow_html=True)

    # Render selected view
    if "01 //" in choice:
        render_overview_view()
    elif "02 //" in choice:
        render_benchmark_view()
    elif "03 //" in choice:
        render_results_view()
    elif "04 //" in choice:
        render_correlation_view()
    elif "05 //" in choice:
        render_sensitivity_view()
    elif "06 //" in choice:
        render_data_view()


if __name__ == "__main__":
    # Check if running within Streamlit CLI
    is_streamlit = (
        "streamlit" in sys.modules
        and hasattr(sys.modules["streamlit"], "_is_running_with_streamlit")
        and sys.modules["streamlit"]._is_running_with_streamlit
    )

    if is_streamlit:
        run_streamlit_app()
    else:
        # User ran python app.py directly -> launch standalone high-speed web server
        from server import run_server
        run_server()
