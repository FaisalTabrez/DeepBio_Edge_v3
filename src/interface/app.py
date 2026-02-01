"""Streamlit dashboard for Global-BioScan visualization and exploration."""

import logging

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Global-BioScan: Deep-Sea Biodiversity Explorer",
    page_icon="🧬",
    layout="wide",
)

logger = logging.getLogger(__name__)


def main():
    """Main Streamlit application."""
    st.title("🧬 Global-BioScan: Deep-Sea Biodiversity Explorer")
    st.markdown(
        """
    An AI-driven pipeline for identifying eukaryotic taxa and assessing 
    biodiversity from deep-sea eDNA datasets.
    """
    )

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "Dashboard",
                "Data Ingestion",
                "Embedding Explorer",
                "Novelty Detection",
                "Diversity Metrics",
                "Configuration",
            ],
        )

    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Ingestion":
        show_ingestion()
    elif page == "Embedding Explorer":
        show_embeddings()
    elif page == "Novelty Detection":
        show_novelty()
    elif page == "Diversity Metrics":
        show_diversity()
    elif page == "Configuration":
        show_config()


def show_dashboard():
    """Main dashboard page."""
    st.header("Dashboard")
    st.info("📊 Summary statistics and key metrics coming soon...")
    # TODO: Display summary cards, charts, etc.


def show_ingestion():
    """Data ingestion page."""
    st.header("Data Ingestion")
    st.info("📤 Upload FASTA files or fetch from OBIS/NCBI...")
    # TODO: File upload, batch creation, etc.


def show_embeddings():
    """Embedding explorer page."""
    st.header("Embedding Explorer")
    st.info("🎨 Interactive UMAP visualization and vector search...")
    # TODO: UMAP plot, similarity search, etc.


def show_novelty():
    """Novelty detection page."""
    st.header("Novel Taxonomic Units")
    st.info("🔍 Discover sequences not in standard databases...")
    # TODO: Novelty results table, taxonomy assignment, etc.


def show_diversity():
    """Diversity metrics page."""
    st.header("Biodiversity Metrics")
    st.info("📈 Alpha/Beta diversity analysis and geographic distribution...")
    # TODO: Diversity charts, heatmaps, etc.


def show_config():
    """Configuration page."""
    st.header("Configuration")
    st.info("⚙️ System settings and advanced options...")
    # TODO: Display config, allow edits, etc.


if __name__ == "__main__":
    main()
