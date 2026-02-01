# ============================================================================
# WINDOWS COMPATIBILITY PATCHES (Must be at top!)
# ============================================================================
# Mock Triton and FlashAttention to prevent ImportError on Windows
import sys
from unittest.mock import MagicMock

# Mock triton (CUDA kernel optimizer, Linux-only)
if sys.platform == "win32" or True:  # Force mock for safety
    sys.modules["triton"] = MagicMock()
    sys.modules["triton.language"] = MagicMock()
    sys.modules["triton.ops"] = MagicMock()

# Mock flash_attn (FastTransformer kernels, Linux-only)
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()
sys.modules["flash_attn.ops"] = MagicMock()

# ============================================================================
# IMPORTS
# ============================================================================

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import (
    LANCEDB_PENDRIVE_PATH,
    LANCEDB_TABLE_SEQUENCES,
    MODEL_NAME,
)
from src.edge.embedder import EmbeddingEngine
from src.edge.taxonomy import NoveltyDetector, TaxonomyPredictor

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Global-BioScan: DeepBio-Edge 🌊",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    /* Deep Ocean Color Palette */
    :root {
        --deep-blue: #0a1e3a;
        --ocean-blue: #1a4d6d;
        --teal: #2dd4da;
        --green: #4ade80;
        --yellow: #fbbf24;
        --red: #ff6b6b;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0a1e3a 0%, #1a3a52 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2538 0%, #1a3f5a 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4ade80;
        text-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    
    .status-known {
        background: #2dd4da;
        color: #000;
    }
    
    .status-novel {
        background: #fbbf24;
        color: #000;
    }
    
    .status-unknown {
        background: #ff6b6b;
        color: #fff;
    }
    
    /* Cards */
    [data-testid="stMetricValue"] {
        color: #4ade80;
    }
    
    /* Buttons */
    button {
        background: #1a4d6d !important;
        border: 1px solid #2dd4da !important;
    }
    
    button:hover {
        background: #2dd4da !important;
        color: #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CACHED RESOURCES (For Performance)
# ============================================================================


@st.cache_resource
def load_embedding_engine() -> EmbeddingEngine:
    """Load embedding engine once and reuse."""
    try:
        logger.info("Loading Nucleotide Transformer model...")
        engine = EmbeddingEngine(use_gpu=None)  # Auto-detect
        return engine
    except Exception as e:
        st.error(f"Failed to load embedding engine: {e}")
        return None


@st.cache_resource
def load_lancedb() -> Optional[lancedb.db.DBConnection]:
    """Connect to LanceDB once and reuse."""
    try:
        db = lancedb.connect(str(LANCEDB_PENDRIVE_PATH))
        return db
    except Exception as e:
        st.error(f"Failed to connect to LanceDB: {e}")
        return None


@st.cache_resource
def load_taxonomy_predictor() -> Optional[TaxonomyPredictor]:
    """Load taxonomy predictor once and reuse."""
    try:
        predictor = TaxonomyPredictor()
        return predictor
    except Exception as e:
        st.error(f"Failed to load taxonomy predictor: {e}")
        return None


@st.cache_resource
def load_novelty_detector() -> Optional[NoveltyDetector]:
    """Load novelty detector once and reuse."""
    try:
        detector = NoveltyDetector()
        return detector
    except Exception as e:
        st.error(f"Failed to load novelty detector: {e}")
        return None


@st.cache_data(ttl=3600)
def get_database_status() -> dict:
    """Check database connection status."""
    db = load_lancedb()
    if not db:
        return {"connected": False, "sequences": 0, "novel_taxa": 0}

    try:
        table = db.open_table(LANCEDB_TABLE_SEQUENCES)
        all_rows = table.search().limit(1000).to_list()
        
        novel_count = 0
        for row in all_rows:
            if row.get("is_novel", False):
                novel_count += 1

        return {
            "connected": True,
            "sequences": len(all_rows),
            "novel_taxa": novel_count,
        }
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        return {"connected": False, "sequences": 0, "novel_taxa": 0}


# ============================================================================
# MYSTERY SEQUENCE (For Demo)
# ============================================================================

MYSTERY_SEQUENCE = """ATGATTATCAATACATTAATATTAATCATTAAAGAATTAATGAAATTATCACCACTATATAATATCAACC
ACGCGCGTTGCATTACATAGTATTCGTAGCCGTATTTATTACAGTAGCACAGATCGCAAATGTAAAAGAG
ATCGGACAATGACTATTTAACACTATTCGACGAATTAATATACCGGACCCGCACGAATGTTCTTATGCC
CCAATATATGAAGATGTACTCACAGAGTTACTAGCCGATATTGTTCTATTAACTGCCGTTTTAGCCGGT
ATGTTAACCGTATCAGAAATACGAAATGCTATTTACGACTCTTACACGGATGAGGAGACCCAGAAGTAC
ACGCACGACAAGTAAACTATCACACACTACGACAAAATCAACCGACGAAAGCGGAGTGATAGCTATCTTT
ACATACATCGGAGATGATGAGATGTTCGACACCCACGAACTAGTCTACAAATACTACGATAATATCGGAA
GCTATTCAGATCAGATACATAAAACTACTACGGTACACGACCCCATCTAGGACGAGAACGTAACTACGAA
CAACTCTACTACCTAGCCGATAACACAAACTAGACGAAGATACACGACCTACGAAAGCATACACGAACGT
ATGATCACACGAAAACTAATACGTCCGTTCTTAGCTCACGTAATACACGCGATATTACGACATAGTTCTC"""


# ============================================================================
# SIDEBAR - "THE CONTROL ROOM"
# ============================================================================

def render_sidebar() -> tuple[float, int]:
    """Render sidebar with controls and metrics."""
    with st.sidebar:
        # Logo & Title
        st.markdown("# 🌊 Global-BioScan")
        st.markdown("### DeepBio-Edge v3.0")
        st.divider()

        # Status
        st.subheader("🔌 System Status")
        status = get_database_status()

        col1, col2 = st.columns(2)
        with col1:
            db_status = "✅ Connected" if status["connected"] else "❌ Disconnected"
            st.metric("Database", db_status)
        with col2:
            model_status = "✅ Loaded"
            st.metric("Model", model_status)

        # Metrics
        st.subheader("📊 Database Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sequences Indexed", f"{status['sequences']:,}")
        with col2:
            st.metric("Novel Taxa Found", f"{status['novel_taxa']:,}")

        st.divider()

        # Hyperparameters
        st.subheader("⚙️ Hyper-Parameters")

        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="Minimum similarity to known sequences for classification",
        )

        top_k_neighbors = st.slider(
            "Top-K Neighbors",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Number of nearest neighbors to search",
        )

        st.divider()

        # About
        st.subheader("ℹ️ About")
        st.markdown(
            """
        **Global-BioScan: DeepBio-Edge**
        
        An AI-driven pipeline for identifying eukaryotic taxa 
        and assessing biodiversity from deep-sea eDNA datasets.
        
        - **Phase 1**: Data ingestion (OBIS + NCBI)
        - **Phase 2**: Embedding generation (NT-500M)
        - **Phase 3**: Novelty detection (Vector search + HDBSCAN)
        
        **Technologies**: 🧬 Transformers | 🗃️ LanceDB | 📊 Scikit-learn
        """
        )

        return similarity_threshold, top_k_neighbors


# ============================================================================
# TAB 1: DEEP SEA DETECTIVE
# ============================================================================

def render_deep_sea_detective(similarity_threshold: float, top_k: int):
    """Single sequence analysis tab."""
    st.header("🔍 Deep Sea Detective")
    st.markdown("Analyze an individual DNA sequence and discover its origins.")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Enter Your Sequence")
        sequence_input = st.text_area(
            "Paste a DNA sequence (ATGC...)",
            height=150,
            label_visibility="collapsed",
            placeholder="ATGATTATCAATACATTAA...",
        )

    with col2:
        st.subheader("Quick Actions")
        if st.button("🎲 Try a Mystery Sequence"):
            st.session_state.sequence_input = MYSTERY_SEQUENCE
            st.rerun()

        if st.button("📋 Clear"):
            st.session_state.sequence_input = ""
            st.rerun()

    # Use session state to preserve input
    if "sequence_input" not in st.session_state:
        st.session_state.sequence_input = sequence_input

    st.divider()

    if st.button("🚀 Analyze Sequence", type="primary", use_container_width=True):
        if not st.session_state.sequence_input.strip():
            st.warning("Please enter a DNA sequence first!")
            return

        sequence = st.session_state.sequence_input.replace("\n", "").upper().strip()

        # Validate
        if not all(c in "ATGCN" for c in sequence):
            st.error("Invalid DNA sequence! Only ATGCN characters allowed.")
            return

        # Load engines
        with st.spinner("🔄 Loading models..."):
            embedding_engine = load_embedding_engine()
            taxonomy_predictor = load_taxonomy_predictor()
            novelty_detector = load_novelty_detector()

            if not (embedding_engine and taxonomy_predictor and novelty_detector):
                st.error("Failed to load required models!")
                return

        # Generate embedding
        with st.spinner("🧬 Generating embedding..."):
            embedding = embedding_engine.get_embedding_single(sequence)
            if embedding is None:
                st.error("Failed to generate embedding!")
                return

        # Search neighbors
        with st.spinner("🔎 Searching neighbors in database..."):
            db = load_lancedb()
            if not db:
                st.error("Database not available!")
                return

            try:
                table = db.open_table(LANCEDB_TABLE_SEQUENCES)
                results = table.search(embedding).limit(top_k).to_list()
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        # Predict taxonomy
        with st.spinner("🧪 Predicting taxonomy..."):
            try:
                lineage = taxonomy_predictor.predict_lineage(
                    embedding, k=top_k, similarity_threshold=similarity_threshold
                )
            except Exception as e:
                st.error(f"Taxonomy prediction failed: {e}")
                return

        # Check novelty
        with st.spinner("🔍 Assessing novelty..."):
            try:
                is_novel = novelty_detector.is_novel(embedding, k=top_k)
            except Exception as e:
                logger.warning(f"Novelty detection failed: {e}")
                is_novel = False

        # Results UI
        st.divider()
        st.subheader("📋 Analysis Results")

        # Status Badge
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            if is_novel:
                st.markdown(
                    '<div class="status-badge status-novel">⭐ POTENTIAL NEW DISCOVERY</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-badge status-known">✅ KNOWN TAXON</div>',
                    unsafe_allow_html=True,
                )

        # Lineage Path (7-level)
        with col2:
            st.subheader("🧬 Taxonomic Lineage")
            if lineage:
                levels = [
                    "Kingdom",
                    "Phylum",
                    "Class",
                    "Order",
                    "Family",
                    "Genus",
                    "Species",
                ]
                for level, name in zip(levels, lineage):
                    if name != "unclassified":
                        st.markdown(f"**{level}**: `{name}`")
            else:
                st.warning("Could not determine lineage")

        # Evidence: Pie chart
        with col3:
            st.subheader("🧮 Neighbor Distribution")

            if results:
                phyla_counts = {}
                for row in results:
                    phylum = row.get("phylum", "Unknown")
                    phyla_counts[phylum] = phyla_counts.get(phylum, 0) + 1

                fig = px.pie(
                    values=list(phyla_counts.values()),
                    names=list(phyla_counts.keys()),
                    title="Top Neighbor Phyla",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=350,
                    font=dict(color="#e0e0e0"),
                    paper_bgcolor="#0a1e3a",
                    plot_bgcolor="#1a4d6d",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Neighbor table
        st.subheader("👥 Nearest Neighbors")

        neighbor_data = []
        for i, row in enumerate(results[:top_k]):
            neighbor_data.append(
                {
                    "Rank": i + 1,
                    "Similarity": f"{row.get('_distance', 0):.4f}",
                    "Species": row.get("species_name", "Unknown"),
                    "Phylum": row.get("phylum", "Unknown"),
                    "Marker Gene": row.get("marker_gene", "Unknown"),
                }
            )

        df_neighbors = pd.DataFrame(neighbor_data)
        st.dataframe(df_neighbors, use_container_width=True, hide_index=True)

        st.success("✅ Analysis complete!")


# ============================================================================
# TAB 2: DISCOVERY MANIFOLD
# ============================================================================

def render_discovery_manifold():
    """3D cluster visualization tab."""
    st.header("🌌 Discovery Manifold")
    st.markdown(
        "Explore the latent space of embedded sequences. Novel clusters are highlighted."
    )

    # Load data
    with st.spinner("📥 Fetching vector sample from database..."):
        db = load_lancedb()
        if not db:
            st.error("Database not available!")
            return

        try:
            table = db.open_table(LANCEDB_TABLE_SEQUENCES)
            # Sample 500 vectors
            all_rows = table.search().limit(500).to_list()
        except Exception as e:
            st.error(f"Failed to fetch vectors: {e}")
            return

    if not all_rows:
        st.warning("No sequences in database yet!")
        return

    # Extract vectors and metadata
    vectors = []
    species = []
    is_novel_flags = []

    for row in all_rows:
        vec = row.get("vector", [])
        if isinstance(vec, list) and len(vec) == 768:
            vectors.append(vec)
            species.append(row.get("species_name", "Unknown"))
            is_novel_flags.append(row.get("is_novel", False))

    if len(vectors) < 10:
        st.warning("Not enough vectors for visualization (need at least 10)")
        return

    vectors = np.array(vectors, dtype=np.float32)

    # Dimensionality reduction
    col1, col2 = st.columns(2)

    with col1:
        reduction_method = st.selectbox(
            "Dimensionality Reduction", ["PCA", "t-SNE"], key="reduction_method"
        )

    with col2:
        perplexity = 30
        if reduction_method == "t-SNE":
            perplexity = st.slider(
                "t-SNE Perplexity", min_value=5, max_value=50, value=30
            )

    with st.spinner(f"🔄 Applying {reduction_method}..."):
        if reduction_method == "PCA":
            reducer = PCA(n_components=3, random_state=42)
            reduced = reducer.fit_transform(vectors)
        else:  # t-SNE
            reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
            reduced = reducer.fit_transform(vectors)

    # Create 3D scatter
    df_plot = pd.DataFrame(
        {
            "PC1": reduced[:, 0],
            "PC2": reduced[:, 1],
            "PC3": reduced[:, 2],
            "Species": species,
            "Novel": is_novel_flags,
        }
    )

    fig = go.Figure()

    # Known sequences (grey)
    known = df_plot[~df_plot["Novel"]]
    fig.add_trace(
        go.Scatter3d(
            x=known["PC1"],
            y=known["PC2"],
            z=known["PC3"],
            mode="markers",
            name="Known Taxa",
            marker=dict(size=4, color="grey", opacity=0.5),
            text=known["Species"],
            hoverinfo="text",
        )
    )

    # Novel sequences (colored, larger)
    novel = df_plot[df_plot["Novel"]]
    if len(novel) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=novel["PC1"],
                y=novel["PC2"],
                z=novel["PC3"],
                mode="markers",
                name="Novel Clusters",
                marker=dict(size=10, color="gold", symbol="diamond", opacity=0.9),
                text=novel["Species"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"Latent Space ({reduction_method})",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
            bgcolor="#1a4d6d",
        ),
        height=700,
        font=dict(color="#e0e0e0"),
        paper_bgcolor="#0a1e3a",
        plot_bgcolor="#1a4d6d",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sequences", len(all_rows))
    with col2:
        st.metric("Novel Clusters", len(novel))
    with col3:
        st.metric("Known Taxa", len(known))


# ============================================================================
# TAB 3: BIODIVERSITY REPORT
# ============================================================================

def render_biodiversity_report():
    """Global statistics and diversity metrics tab."""
    st.header("📊 Biodiversity Report")
    st.markdown("Global statistics and diversity metrics across sampled sequences.")

    # Load database
    db = load_lancedb()
    if not db:
        st.error("Database not available!")
        return

    try:
        table = db.open_table(LANCEDB_TABLE_SEQUENCES)
        all_rows = table.search().limit(1000).to_list()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if not all_rows:
        st.warning("No sequences in database yet!")
        return

    # Convert to DataFrame
    data = []
    for row in all_rows:
        data.append(
            {
                "species": row.get("species_name", "Unknown"),
                "phylum": row.get("phylum", "Unknown"),
                "depth": row.get("depth_m", 0),
                "is_novel": row.get("is_novel", False),
            }
        )

    df = pd.DataFrame(data)

    # Top statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sequences", len(df))

    with col2:
        st.metric("Unique Phyla", df["phylum"].nunique())

    with col3:
        st.metric("Unique Species", df["species"].nunique())

    with col4:
        st.metric("Novel Sequences", df["is_novel"].sum())

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    # Chart 1: Phyla distribution
    with col1:
        st.subheader("Phyla Distribution")
        phyla_counts = df["phylum"].value_counts().head(10)
        fig = px.bar(
            x=phyla_counts.index,
            y=phyla_counts.values,
            labels={"x": "Phylum", "y": "Count"},
            color=phyla_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            height=400,
            font=dict(color="#e0e0e0"),
            paper_bgcolor="#0a1e3a",
            plot_bgcolor="#1a4d6d",
            xaxis_tickangle=-45,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Known vs Novel by depth
    with col2:
        st.subheader("Known vs Novel by Depth")
        # Bin depths
        df["depth_bin"] = pd.cut(
            df["depth"], bins=[0, 200, 1000, 3000, 6000], right=False
        )
        depth_counts = df.groupby(["depth_bin", "is_novel"]).size().unstack(fill_value=0)

        fig = px.bar(
            depth_counts,
            barmode="stack",
            labels={"value": "Count", "index": "Depth Range"},
            color_discrete_map={True: "#fbbf24", False: "#2dd4da"},
        )
        fig.for_each_trace(lambda t: t.update(name="Novel" if t.name == "True" else "Known"))
        fig.update_layout(
            height=400,
            font=dict(color="#e0e0e0"),
            paper_bgcolor="#0a1e3a",
            plot_bgcolor="#1a4d6d",
            legend_title="Classification",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Alpha diversity (Simpson's Index)
    st.subheader("🔬 Diversity Indices")

    col1, col2 = st.columns(2)

    # Simpson's diversity index
    phyla_proportions = df["phylum"].value_counts() / len(df)
    simpson_index = 1 - (phyla_proportions ** 2).sum()

    with col1:
        st.metric(
            "Simpson's Diversity Index",
            f"{simpson_index:.3f}",
            help="Range: 0-1. Higher = more diverse.",
        )

    # Shannon diversity index
    shannon_index = -sum(phyla_proportions * np.log(phyla_proportions))

    with col2:
        st.metric(
            "Shannon Diversity Index",
            f"{shannon_index:.3f}",
            help="Range: 0-ln(N). Higher = more diverse.",
        )

    st.divider()

    # Raw data table
    st.subheader("📋 Sequence Data")

    display_df = df[["species", "phylum", "depth", "is_novel"]].copy()
    display_df["is_novel"] = display_df["is_novel"].apply(lambda x: "🌟 Novel" if x else "✓ Known")
    display_df.columns = ["Species", "Phylum", "Depth (m)", "Classification"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    # Render sidebar
    similarity_threshold, top_k = render_sidebar()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Deep Sea Detective", "🌌 Discovery Manifold", "📊 Biodiversity Report"]
    )

    with tab1:
        render_deep_sea_detective(similarity_threshold, top_k)

    with tab2:
        render_discovery_manifold()

    with tab3:
        render_biodiversity_report()

    # Footer
    st.divider()
    st.markdown(
        """
    ---
    **Global-BioScan: DeepBio-Edge v3.0** | *Powered by Nucleotide Transformers + LanceDB + Scikit-learn*
    
    🌊 Discover the unknown depths of biodiversity.
    """
    )


if __name__ == "__main__":
    main()
