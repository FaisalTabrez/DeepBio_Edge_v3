"""Professional Streamlit interface for Global-BioScan Genomic Analysis Platform.

Scientific-grade biotechnological interface for taxonomic inference and ecological analysis.
Supports standard bioinformatics file formats: FASTA, FASTQ, CSV, TXT, Parquet.
"""
# pyright: reportOptionalMemberAccess=false
# ============================================================================
# WINDOWS COMPATIBILITY PATCHES (Must be at top!)
# ============================================================================
import sys
from pathlib import Path
from unittest.mock import MagicMock
import io
from typing import Optional, List, Dict, Any, Tuple

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Mock triton and all its submodules (CUDA kernel optimizer, Linux-only)
if sys.platform == "win32" or True:
    triton_mock = MagicMock()
    sys.modules["triton"] = triton_mock
    sys.modules["triton.language"] = triton_mock
    sys.modules["triton.ops"] = triton_mock
    sys.modules["triton.backends"] = triton_mock
    sys.modules["triton.backends.compiler"] = triton_mock
    sys.modules["triton.compiler"] = triton_mock
    sys.modules["triton.compiler.compiler"] = triton_mock
    sys.modules["triton.runtime"] = triton_mock

# Mock flash_attn
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()
sys.modules["flash_attn.ops"] = MagicMock()

# ============================================================================
# IMPORTS
# ============================================================================

import logging
from datetime import datetime

import lancedb
import numpy as np
import pandas as pd

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except ImportError:
    px = None  # type: ignore
    go = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Biopython for sequence parsing
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from src.config import (
    LANCEDB_PENDRIVE_PATH,
    LANCEDB_TABLE_SEQUENCES,
    MODEL_NAME,
)

# Conditional imports to avoid transformers/torch loading issues
try:
    from src.edge.embedder import EmbeddingEngine
    EMBEDDING_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    EmbeddingEngine = None  # type: ignore
    EMBEDDING_AVAILABLE = False

from src.edge.taxonomy import NoveltyDetector, TaxonomyPredictor

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Global-BioScan: Genomic Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Professional Scientific Palette */
    .stApp {
        background-color: #0a1929;
    }
    .stSidebar {
        background-color: #132f4c;
    }
    h1, h2, h3 {
        color: #66d9ef;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 4px;
        font-weight: 600;
    }
    .stMetric {
        background-color: #1a2332;
        padding: 12px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# FILE PARSING UTILITIES
# ============================================================================

def parse_bio_file(uploaded_file) -> List[Dict[str, str]]:
    """Parse bioinformatics files into standardized sequence records.
    
    Supports: FASTA, FASTQ, CSV, TXT, Parquet
    
    Returns:
        List of dicts with 'id' and 'sequence' keys
    """
    sequences = []
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension in ['fasta', 'fa', 'fna']:
            if not BIOPYTHON_AVAILABLE:
                st.error("BioPython required for FASTA parsing. Install with: pip install biopython")
                return []
            
            content = uploaded_file.getvalue().decode('utf-8')
            fasta_io = io.StringIO(content)
            
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences.append({
                    'id': record.id,
                    'sequence': str(record.seq).upper()
                })
        
        elif file_extension in ['fastq', 'fq']:
            if not BIOPYTHON_AVAILABLE:
                st.error("BioPython required for FASTQ parsing. Install with: pip install biopython")
                return []
            
            content = uploaded_file.getvalue().decode('utf-8')
            fastq_io = io.StringIO(content)
            
            for record in SeqIO.parse(fastq_io, "fastq"):
                sequences.append({
                    'id': record.id,
                    'sequence': str(record.seq).upper()
                })
        
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            
            # Try to find sequence columns
            seq_col = None
            id_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'seq' in col_lower and seq_col is None:
                    seq_col = col
                if 'id' in col_lower or 'name' in col_lower:
                    id_col = col
            
            if seq_col is None:
                st.error("Could not find sequence column. Expected column with 'seq' in name.")
                return []
            
            if id_col is None:
                id_col = df.columns[0]  # Use first column as ID
            
            for idx, row in df.iterrows():
                sequences.append({
                    'id': str(row[id_col]) if pd.notna(row[id_col]) else f"seq_{idx+1}",
                    'sequence': str(row[seq_col]).upper().replace('\n', '').replace(' ', '')
                })
        
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
            
            # Try to find sequence columns
            seq_col = None
            id_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'seq' in col_lower and seq_col is None:
                    seq_col = col
                if 'id' in col_lower or 'name' in col_lower:
                    id_col = col
            
            if seq_col is None:
                st.error("Could not find sequence column. Expected column with 'seq' in name.")
                return []
            
            if id_col is None:
                id_col = df.columns[0]
            
            for idx, row in df.iterrows():
                sequences.append({
                    'id': str(row[id_col]) if pd.notna(row[id_col]) else f"seq_{idx+1}",
                    'sequence': str(row[seq_col]).upper().replace('\n', '').replace(' ', '')
                })
        
        elif file_extension == 'txt':
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Assume plain text with one sequence per line (or single sequence)
            if len(lines) == 1:
                sequences.append({
                    'id': 'input_sequence_1',
                    'sequence': lines[0].strip().upper().replace(' ', '')
                })
            else:
                for idx, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments
                        sequences.append({
                            'id': f'seq_{idx+1}',
                            'sequence': line.upper().replace(' ', '')
                        })
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []
        
        # Validate sequences
        valid_sequences = []
        for seq_record in sequences:
            seq = seq_record['sequence']
            if seq and all(c in 'ATGCNRYSWKMBDHV' for c in seq):
                valid_sequences.append(seq_record)
            else:
                st.warning(f"Skipping invalid sequence ID: {seq_record['id']}")
        
        return valid_sequences
    
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        logger.error(f"File parsing error: {str(e)}", exc_info=True)
        return []


def export_darwin_core_csv(results: List[Dict[str, Any]]) -> str:
    """Export results as Darwin Core standard CSV format."""
    df = pd.DataFrame(results)
    
    # Rename to Darwin Core terms
    darwin_core_mapping = {
        'id': 'occurrenceID',
        'sequence': 'associatedSequences',
        'predicted_lineage': 'scientificName',
        'confidence': 'identificationRemarks',
        'status': 'occurrenceStatus'
    }
    
    for old_col, new_col in darwin_core_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Add metadata
    df['basisOfRecord'] = 'MachineObservation'
    df['identificationMethod'] = 'Nucleotide Transformer Deep Learning Model'
    df['dateIdentified'] = datetime.now().strftime('%Y-%m-%d')
    
    return df.to_csv(index=False)


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def load_embedding_engine() -> Optional[EmbeddingEngine]:  # type: ignore
    """Load embedding engine once and reuse."""
    if not EMBEDDING_AVAILABLE:
        logger.warning("EmbeddingEngine not available (transformers import failed)")
        return None
    try:
        with st.status("Initializing Nucleotide Transformer (500M parameters)..."):
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
        logger.info(f"Connected to LanceDB at {LANCEDB_PENDRIVE_PATH}")
        return db
    except Exception as e:
        st.error(f"Failed to connect to LanceDB: {e}")
        return None


@st.cache_resource
def load_taxonomy_predictor() -> Optional[TaxonomyPredictor]:
    """Load taxonomy predictor once and reuse."""
    try:
        predictor = TaxonomyPredictor(
            db_path=str(LANCEDB_PENDRIVE_PATH),
            table_name=LANCEDB_TABLE_SEQUENCES,
            enable_taxonkit=False,
        )
        return predictor
    except Exception as e:
        st.error(f"Failed to load taxonomy predictor: {e}")
        return None


@st.cache_resource
def load_novelty_detector() -> Optional[NoveltyDetector]:
    """Load novelty detector once and reuse."""
    try:
        detector = NoveltyDetector(db_path=str(LANCEDB_PENDRIVE_PATH))
        return detector
    except Exception as e:
        st.error(f"Failed to load novelty detector: {e}")
        return None


@st.cache_data(ttl=3600)
def get_database_status() -> Dict[str, Any]:
    """Get current database statistics."""
    try:
        db = load_lancedb()
        if db is None:
            return {"sequences": 0, "novel_taxa": 0, "status": "disconnected"}
        
        table = db.open_table(LANCEDB_TABLE_SEQUENCES)
        count = table.count_rows()
        
        return {
            "sequences": count,
            "novel_taxa": 0,  # Placeholder
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        return {"sequences": 0, "novel_taxa": 0, "status": "error"}


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render professional sidebar with system metrics."""
    with st.sidebar:
        st.markdown("# 🧬 Global-BioScan")
        st.markdown("### Genomic Analysis Platform v3.0")
        st.divider()
        
        # System Status
        st.subheader("🔌 System Status")
        status = get_database_status()
        
        db_status = "🟢 Online" if status["status"] == "connected" else "🔴 Offline"
        model_status = "🟢 Ready" if EMBEDDING_AVAILABLE else "🟡 Unavailable"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Database", db_status)
        with col2:
            st.metric("ML Model", model_status)
        
        # Database Metrics
        st.subheader("📊 Database Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sequences Indexed", f"{status['sequences']:,}")
        with col2:
            st.metric("Novel Taxa Detected", f"{status['novel_taxa']:,}")
        
        st.divider()
        
        # Inference Parameters
        st.subheader("⚙️ Inference Parameters")
        
        similarity_threshold = st.slider(
            "Identity Confidence Threshold (σ)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01,
            help="Minimum cosine similarity for taxonomic assignment"
        )
        
        top_k_neighbors = st.slider(
            "K-Nearest Neighbors",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="Number of nearest vectors to retrieve from database"
        )
        
        st.divider()
        
        # Model Architecture Info
        st.subheader("🧠 Model Architecture")
        st.markdown("""
        **Model:** Nucleotide Transformer  
        **Parameters:** 500M (500 Million)  
        **Embedding Dimension:** 768  
        **Context Window:** 6,000 nucleotides  
        **Pre-training:** Multi-species genomic corpus
        """)
        
        st.subheader("💾 Backend Infrastructure")
        st.markdown("""
        **Vector Store:** LanceDB (Disk-Native)  
        **Index Type:** IVF-PQ (Inverted File Index)  
        **Storage:** 32GB Pendrive (Edge Deployment)  
        **Similarity Metric:** Cosine Distance
        """)
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **Global-BioScan** is a deep learning-powered platform for 
            taxonomic inference from environmental DNA (eDNA) sequences.
            
            **Citation:**  
            DeepBio-Edge: Large-scale Biodiversity Monitoring via  
            Foundation Models on Edge Devices (2026)
            
            **Developed by:** Global-BioScan Consortium  
            **License:** MIT
            """
        )
        
        return similarity_threshold, top_k_neighbors


# ============================================================================
# TAB 1: TAXONOMIC INFERENCE ENGINE
# ============================================================================

def render_taxonomic_inference_engine(similarity_threshold: float, top_k_neighbors: int):
    """Professional taxonomic inference interface with batch processing."""
    st.header("🔬 Taxonomic Inference Engine")
    st.markdown("Execute deep learning-based taxonomic classification on DNA sequences.")
    
    # Processing Mode Selection
    col1, col2 = st.columns([3, 1])
    with col1:
        processing_mode = st.radio(
            "Processing Mode",
            ["Single Sequence", "Batch Processing"],
            horizontal=True
        )
    with col2:
        st.markdown("")  # Spacer
    
    st.divider()
    
    # File Upload or Text Input
    st.subheader("📂 Genetic Input Configuration")
    
    input_method = st.radio(
        "Input Method",
        ["File Upload", "Manual Entry"],
        horizontal=True
    )
    
    sequences_to_process = []
    
    if input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload Sequence File",
            type=['fasta', 'fa', 'fna', 'fastq', 'fq', 'csv', 'txt', 'parquet'],
            help="Supported formats: FASTA, FASTQ, CSV, TXT, Parquet"
        )
        
        if uploaded_file is not None:
            with st.status("Parsing bioinformatics file..."):
                sequences_to_process = parse_bio_file(uploaded_file)
            
            if sequences_to_process:
                st.success(f"✅ Parsed {len(sequences_to_process)} valid sequences from file")
                
                # Preview
                with st.expander("📋 Sequence Preview"):
                    preview_df = pd.DataFrame(sequences_to_process[:10])  # Show first 10
                    preview_df['sequence_length'] = preview_df['sequence'].str.len()
                    st.dataframe(preview_df, use_container_width=True)
    
    else:  # Manual Entry
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sequence_input = st.text_area(
                "DNA Sequence (ATGCN)",
                height=120,
                placeholder="ATGCGATCGTAGCTACGTACG...",
                help="Enter nucleotide sequence using IUPAC codes"
            )
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("📋 Load Reference Template"):
                reference_seq = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
                st.session_state.sequence_input = reference_seq
                st.rerun()
            
            if st.button("🗑️ Clear"):
                st.session_state.sequence_input = ""
                st.rerun()
        
        if "sequence_input" not in st.session_state:
            st.session_state.sequence_input = sequence_input
        
        if sequence_input.strip():
            sequences_to_process = [{
                'id': 'manual_input',
                'sequence': sequence_input.strip().upper().replace('\n', '').replace(' ', '')
            }]
    
    st.divider()
    
    # Execute Inference Button
    if st.button("🚀 Execute Inference", type="primary", use_container_width=True):
        if not sequences_to_process:
            st.warning("⚠️ No sequences to process. Please upload a file or enter a sequence.")
            return
        
        # Load resources
        engine = load_embedding_engine()
        predictor = load_taxonomy_predictor()
        
        if engine is None and processing_mode == "Batch Processing":
            st.error("🔴 Embedding engine unavailable. Cannot perform batch processing.")
            return
        
        if predictor is None:
            st.error("🔴 Taxonomy predictor unavailable.")
            return
        
        # Process sequences
        results = []
        
        if processing_mode == "Single Sequence" or len(sequences_to_process) == 1:
            # Single sequence mode
            seq_record = sequences_to_process[0]
            
            with st.status("Processing sequence...") as status_container:
                status_container.update(label="Validating sequence format...")
                
                # Validate
                sequence = seq_record['sequence']
                valid_chars = set('ATGCNRYSWKMBDHV')
                if not all(c in valid_chars for c in sequence):
                    st.error("❌ Invalid DNA sequence. Only IUPAC nucleotide codes allowed.")
                    return
                
                status_container.update(label="Generating embeddings...")
                
                # Get embedding
                if engine:
                    try:
                        vector = engine.get_embedding_single(sequence)
                    except Exception as e:
                        st.error(f"Embedding error: {e}")
                        return
                else:
                    st.warning("Using mock embeddings (model unavailable)")
                    from src.benchmarks.mock_community import deterministic_vector
                    vector = deterministic_vector(sequence)
                
                status_container.update(label="Searching vector database...")
                
                # Predict taxonomy
                neighbors = predictor.search_neighbors(vector.tolist(), k=top_k_neighbors)
                prediction = predictor.predict_lineage(neighbors)
                
                status_container.update(label="✅ Inference complete", state="complete")
            
            # Display result
            st.divider()
            st.subheader("📊 Taxonomic Classification Result")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Sequence ID:** `{seq_record['id']}`")
                st.markdown(f"**Length:** {len(sequence)} bp")
                st.markdown(f"**Predicted Lineage:**")
                st.code(prediction.lineage, language="")
                
            with col2:
                confidence_color = "🟢" if prediction.confidence > 0.9 else "🟡" if prediction.confidence > 0.7 else "🔴"
                st.metric("Confidence Score", f"{prediction.confidence:.3f}", delta=confidence_color)
                st.metric("Classification Status", prediction.status)
            
            # Nearest Neighbors Table
            st.subheader("🔍 K-Nearest Reference Sequences")
            neighbors_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Reference ID': n.get('sequence_id', 'unknown'),
                    'Lineage': n.get('lineage', 'unknown'),
                    'Similarity': f"{n.get('distance', 0):.4f}"
                }
                for i, n in enumerate(neighbors[:10])
            ])
            st.dataframe(neighbors_df, use_container_width=True)
        
        else:
            # Batch processing mode
            st.subheader("⚙️ Batch Processing Pipeline")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Batch embed all sequences
            status_text.text("Stage 1/3: Generating embeddings (vectorized)...")
            
            if engine:
                try:
                    sequences = [s['sequence'] for s in sequences_to_process]
                    embeddings = engine.get_embeddings(sequences)
                except Exception as e:
                    st.error(f"Batch embedding error: {e}")
                    return
            else:
                from src.benchmarks.mock_community import build_mock_embeddings
                from src.benchmarks.mock_community import MockCommunityEntry
                
                mock_entries = [
                    MockCommunityEntry(s['id'], s['sequence'], "", "")
                    for s in sequences_to_process
                ]
                embeddings = build_mock_embeddings(mock_entries)
            
            progress_bar.progress(33)
            status_text.text("Stage 2/3: Searching vector database...")
            
            # Process each sequence
            for idx, (seq_record, embedding) in enumerate(zip(sequences_to_process, embeddings)):
                neighbors = predictor.search_neighbors(embedding.tolist(), k=top_k_neighbors)
                prediction = predictor.predict_lineage(neighbors)
                
                results.append({
                    'id': seq_record['id'],
                    'sequence': seq_record['sequence'][:50] + '...' if len(seq_record['sequence']) > 50 else seq_record['sequence'],
                    'sequence_length': len(seq_record['sequence']),
                    'predicted_lineage': prediction.lineage,
                    'confidence': prediction.confidence,
                    'status': prediction.status
                })
                
                progress_bar.progress(33 + int((idx + 1) / len(sequences_to_process) * 67))
            
            status_text.text("Stage 3/3: Generating summary report...")
            progress_bar.progress(100)
            
            # Display batch results
            st.divider()
            st.subheader("📊 Batch Inference Summary")
            
            results_df = pd.DataFrame(results)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sequences", len(results))
            with col2:
                high_conf = len(results_df[results_df['confidence'] > 0.9])
                st.metric("High Confidence", high_conf)
            with col3:
                known = len(results_df[results_df['status'] == 'KNOWN'])
                st.metric("Known Taxa", known)
            with col4:
                novel = len(results_df[results_df['status'] == 'NOVEL_CANDIDATE'])
                st.metric("Novel Candidates", novel)
            
            # Results table
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
            
            # Export button
            csv_data = export_darwin_core_csv(results)
            st.download_button(
                label="📥 Download Darwin Core CSV",
                data=csv_data,
                file_name=f"bioscan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            
            st.success(f"✅ Batch inference complete: {len(results)} sequences processed")


# ============================================================================
# TAB 2: LATENT SPACE ANALYSIS
# ============================================================================

def render_latent_space_analysis():
    """Visualize embedding space with dimensionality reduction."""
    st.header("🗺️ Latent Space Analysis")
    st.markdown("Interactive visualization of high-dimensional genomic embeddings.")
    
    db = load_lancedb()
    if db is None:
        st.error("Database connection unavailable.")
        return
    
    try:
        table = db.open_table(LANCEDB_TABLE_SEQUENCES)
        count = table.count_rows()
        
        if count == 0:
            st.warning("⚠️ No sequences in database yet.")
            return
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sample size
        max_samples = min(count, 10000)
        sample_size = st.slider(
            "Sample Size",
            min_value=100,
            max_value=max_samples,
            value=min(1000, max_samples),
            step=100
        )
    
    with col2:
        # Dimensionality reduction method
        reduction_method = st.selectbox(
            "Dimensionality Reduction",
            ["t-SNE", "PCA", "UMAP"],
            help="Algorithm for projecting 768D embeddings to 3D"
        )
    
    with st.status(f"Loading {sample_size} vectors from database..."):
        try:
            # Sample data
            data = table.to_pandas()
            
            if len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
            
            # Extract embeddings
            if 'vector' in data.columns:
                embeddings = np.array(data['vector'].tolist())
            elif 'embedding' in data.columns:
                embeddings = np.array(data['embedding'].tolist())
            else:
                st.error("No embedding column found in database")
                return
            
            # Perform dimensionality reduction
            if reduction_method == "t-SNE":
                reducer = TSNE(n_components=3, random_state=42, perplexity=30)
            elif reduction_method == "PCA":
                reducer = PCA(n_components=3, random_state=42)
            else:  # UMAP
                st.warning("UMAP not available. Falling back to t-SNE.")
                reducer = TSNE(n_components=3, random_state=42, perplexity=30)
            
            embeddings_3d = reducer.fit_transform(embeddings)
            
            # Create visualization
            if go is None:
                st.error("Plotly not available. Cannot render 3D plot.")
                return
            
            # Extract taxonomy for coloring
            if 'lineage' in data.columns:
                # Extract phylum (second level) for coloring
                data['phylum'] = data['lineage'].apply(
                    lambda x: x.split(';')[1] if ';' in str(x) and len(x.split(';')) > 1 else 'Unknown'
                )
                color_by = data['phylum']
            else:
                color_by = 'blue'
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=embeddings_3d[:, 0],
                    y=embeddings_3d[:, 1],
                    z=embeddings_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color_by if isinstance(color_by, str) else range(len(embeddings_3d)),
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    text=data.get('sequence_id', 'unknown'),
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=f"{reduction_method} Projection of Genomic Embedding Space",
                scene=dict(
                    xaxis_title=f"{reduction_method} Component 1",
                    yaxis_title=f"{reduction_method} Component 2",
                    zaxis_title=f"{reduction_method} Component 3",
                    bgcolor='#0a1929'
                ),
                paper_bgcolor='#0a1929',
                plot_bgcolor='#0a1929',
                font=dict(color='#ffffff'),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vectors Visualized", len(embeddings_3d))
            with col2:
                st.metric("Original Dimensionality", embeddings.shape[1])
            with col3:
                st.metric("Reduced Dimensionality", 3)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            logger.error(f"Latent space visualization error: {e}", exc_info=True)


# ============================================================================
# TAB 3: ECOLOGICAL COMPOSITION
# ============================================================================

def render_ecological_composition():
    """Display biodiversity metrics and ecological analysis."""
    st.header("🌿 Ecological Composition Analysis")
    st.markdown("Comprehensive biodiversity assessment and taxonomic distribution.")
    
    db = load_lancedb()
    if db is None:
        st.error("Database connection unavailable.")
        return
    
    try:
        table = db.open_table(LANCEDB_TABLE_SEQUENCES)
        data = table.to_pandas()
        
        if len(data) == 0:
            st.warning("⚠️ No ecological data available yet.")
            return
        
        if 'lineage' not in data.columns:
            st.warning("⚠️ Lineage information not available in database.")
            return
        
        # Parse lineages
        def parse_lineage(lineage_str):
            parts = str(lineage_str).split(';')
            return {
                'kingdom': parts[0] if len(parts) > 0 else 'Unknown',
                'phylum': parts[1] if len(parts) > 1 else 'Unknown',
                'class': parts[2] if len(parts) > 2 else 'Unknown',
                'order': parts[3] if len(parts) > 3 else 'Unknown',
                'family': parts[4] if len(parts) > 4 else 'Unknown',
                'genus': parts[5] if len(parts) > 5 else 'Unknown',
                'species': parts[6] if len(parts) > 6 else 'Unknown',
            }
        
        lineage_parsed = data['lineage'].apply(parse_lineage)
        for rank in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            data[rank] = lineage_parsed.apply(lambda x: x[rank])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sequences", len(data))
        with col2:
            unique_species = data['species'].nunique()
            st.metric("Unique Species", unique_species)
        with col3:
            unique_genera = data['genus'].nunique()
            st.metric("Unique Genera", unique_genera)
        with col4:
            unique_phyla = data['phylum'].nunique()
            st.metric("Unique Phyla", unique_phyla)
        
        st.divider()
        
        # Taxonomic Distribution Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Phylum Distribution")
            
            phylum_counts = data['phylum'].value_counts().head(10)
            
            if px:
                fig = px.bar(
                    x=phylum_counts.index,
                    y=phylum_counts.values,
                    labels={'x': 'Phylum', 'y': 'Sequence Count'},
                    title='Top 10 Phyla by Abundance'
                )
                fig.update_layout(
                    paper_bgcolor='#0a1929',
                    plot_bgcolor='#132f4c',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(phylum_counts)
        
        with col2:
            st.subheader("🧬 Class Distribution")
            
            class_counts = data['class'].value_counts().head(10)
            
            if px:
                fig = px.bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    labels={'x': 'Class', 'y': 'Sequence Count'},
                    title='Top 10 Classes by Abundance'
                )
                fig.update_layout(
                    paper_bgcolor='#0a1929',
                    plot_bgcolor='#132f4c',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(class_counts)
        
        st.divider()
        
        # Detailed Taxonomy Table
        st.subheader("🔬 Taxonomic Inventory")
        
        taxonomy_summary = data.groupby(['kingdom', 'phylum', 'class', 'order']).size().reset_index(name='count')
        taxonomy_summary = taxonomy_summary.sort_values('count', ascending=False)
        
        st.dataframe(
            taxonomy_summary,
            use_container_width=True,
            height=400
        )
        
    except Exception as e:
        st.error(f"Ecological analysis error: {str(e)}")
        logger.error(f"Ecological composition error: {e}", exc_info=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Render sidebar and get parameters
    similarity_threshold, top_k_neighbors = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔬 Taxonomic Inference Engine",
        "🗺️ Latent Space Analysis",
        "🌿 Ecological Composition"
    ])
    
    with tab1:
        render_taxonomic_inference_engine(similarity_threshold, top_k_neighbors)
    
    with tab2:
        render_latent_space_analysis()
    
    with tab3:
        render_ecological_composition()


if __name__ == "__main__":
    main()
