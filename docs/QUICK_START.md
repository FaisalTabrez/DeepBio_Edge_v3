# Global-BioScan Quick Start Guide

## 🚀 Launch Application

```bash
cd "c:\Volume D\DeepBio_Edge_v3"
.venv/Scripts/python -m streamlit run src/interface/app.py
```

Access at: **http://localhost:8501**

---

## 📋 Quick Workflows

### 1️⃣ Analyze a Single Sequence (1 minute)

1. Open **Taxonomic Inference Engine** tab
2. Choose **"Manual Entry"**
3. Paste your DNA sequence:
   ```
   ATGCGATCGATCGATCGATCGATCGATCGATCG
   ```
4. Click **"Execute Inference"** 🚀
5. View results:
   - Predicted taxonomy
   - Confidence score
   - Similar reference sequences

### 2️⃣ Batch Process Multiple Sequences (2-5 minutes)

1. Open **Taxonomic Inference Engine** tab
2. Select **"Batch Processing"** mode
3. Upload your file (FASTA/FASTQ/CSV)
4. Wait for parsing ✅
5. Click **"Execute Inference"** 🚀
6. Download results as **Darwin Core CSV** 📥

### 3️⃣ Visualize Embedding Space (3 minutes)

1. Open **Latent Space Analysis** tab
2. Set sample size (start with 500)
3. Choose **t-SNE** reduction
4. Explore 3D visualization:
   - Rotate: Click + drag
   - Zoom: Scroll wheel
   - Hover: See sequence IDs

### 4️⃣ View Biodiversity Metrics (1 minute)

1. Open **Ecological Composition** tab
2. Review metrics:
   - Species count
   - Genus count
   - Phylum distribution
3. Examine charts and tables

---

## 📁 Supported File Formats

| Format   | Extensions              | Use Case                    |
|----------|-------------------------|-----------------------------|
| FASTA    | `.fasta`, `.fa`, `.fna` | Standard sequence format    |
| FASTQ    | `.fastq`, `.fq`         | Sequences with quality      |
| CSV      | `.csv`                  | Tabular data                |
| TXT      | `.txt`                  | Plain text sequences        |
| Parquet  | `.parquet`              | Large datasets (>1GB)       |

---

## 🔧 Sidebar Settings

### Identity Confidence Threshold (σ)
- **0.9-1.0**: High stringency (conservative)
- **0.7-0.9**: Moderate stringency (balanced)
- **0.5-0.7**: Low stringency (permissive)

### K-Nearest Neighbors
- **5-10**: Standard taxonomic inference
- **20-50**: Consensus-based classification
- **1-3**: Rapid screening

---

## 📊 Understanding Results

### Confidence Scores
- 🟢 **>0.9**: High confidence - Reliable identification
- 🟡 **0.7-0.9**: Moderate confidence - Verify with experts
- 🔴 **<0.7**: Low confidence - Possible novel taxa

### Classification Status
- **KNOWN**: Match to reference database
- **NOVEL_CANDIDATE**: Potential new species
- **AMBIGUOUS**: Multiple conflicting matches

---

## 🎯 Best Practices

1. **Sequence Quality:**
   - Use sequences >100 bp for reliable results
   - Trim adapter sequences before upload
   - Remove low-quality regions

2. **Batch Processing:**
   - Start with <500 sequences for testing
   - Use FASTA for DNA-only data
   - Use CSV for metadata-rich datasets

3. **Visualization:**
   - Use t-SNE for cluster exploration
   - Use PCA for rapid overview
   - Reduce sample size if slow (<1000 vectors)

4. **Data Export:**
   - Always download Darwin Core CSV for archiving
   - Include metadata in original upload (CSV/Parquet)
   - Save high-confidence results separately

---

## 🐛 Common Issues

### Problem: File upload fails
- Check file format extension matches content
- Verify CSV has "sequence" or "seq" column
- Ensure file size <100MB

### Problem: Low confidence scores
- Sequence too short (<100 bp)
- Novel taxa not in database
- Poor sequence quality

### Problem: Slow inference
- Large batch size (reduce to <200)
- Database connection latency
- CPU-only mode (GPU recommended)

---

## 📞 Getting Help

- Check: `docs/PROFESSIONAL_INTERFACE.md` for details
- Demo files: `data/demo/sample_sequences.fasta`
- Test database: `tests/test_lancedb/`

---

## 🎓 Example: Marine eDNA Analysis

**Scenario:** You collected 50 water samples and sequenced COI barcodes.

**Steps:**
1. Prepare CSV file:
   ```csv
   sequence_id,sequence,location,depth_m
   SITE01_001,ATGCG...,Pacific Ocean,100
   SITE01_002,GCTAG...,Pacific Ocean,100
   ```

2. Upload to **Taxonomic Inference Engine** (Batch mode)

3. Review results:
   - 45 known species
   - 5 novel candidates (confidence <0.7)

4. Download Darwin Core CSV

5. Submit novel candidates to **BOLD Systems** for expert validation

6. Visualize in **Latent Space Analysis** to see:
   - Marine vs. terrestrial clusters
   - Phylum-level separation

---

**Happy Analyzing! 🧬**
