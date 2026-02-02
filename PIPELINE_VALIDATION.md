# Pipeline Validation Whitepaper (CMLRE Mission)

## Purpose

This document defines the biological ground truth and validation protocol for the GlobalBioScan pipeline. The goal is to prove the system performs reliably on **Knowns** before claiming discovery of **Unknowns**.

The validation framework uses mock communities with verified lineages, controlled mutation stress tests, vector consistency checks (Cloud vs. Edge), storage performance profiling, and false discovery rate testing.

---

## Biological Ground Truth

### Mock Community Composition

The mock community is a curated set of **50 sequences** representing diverse biological groups:

- **Mammals (12)** — Homo sapiens, Mus musculus, Bos taurus, Canis lupus, Felis catus, etc.
- **Fish (12)** — Gadus morhua, Salmo salar, Danio rerio, Thunnus albacares, etc.
- **Plankton (13)** — Copepods, diatoms, cyanobacteria, and green algae
- **Fungi (13)** — Yeasts, molds, and basidiomycetes

Each entry is labeled with a **7-level NCBI-style lineage**:

```
Kingdom;Phylum;Class;Order;Family;Genus;Species
```

The dataset is stored at:

- [data/test/mock_community.fasta](data/test/mock_community.fasta)

---

## Validation Pipeline

The QC suite runs the full pipeline on mock communities:

1. **DNAEmbedder** — Nucleotide Transformer embeddings (768-dim)
2. **BioDB (LanceDB)** — Vector storage and retrieval (pendrive-based)
3. **TaxonomyPredictor** — K-NN consensus lineage prediction

**Key output:** Precision, Recall, F1-score, and Taxonomic Depth Accuracy.

---

## Metrics & Acceptance Criteria

### 1. Taxonomic Precision / Recall / F1

- **Precision**: Fraction of predicted lineages that are correct
- **Recall**: Fraction of ground-truth lineages recovered
- **F1-score**: Harmonic mean of precision and recall

**Acceptance Target**: F1 ≥ **0.85** on mock community.

### 2. Taxonomic Depth Accuracy

For each prediction, measure how deep the lineage is correct:

- Correct at **Phylum**? **Class**? **Genus**? **Species**?

**Depth Accuracy Score** = number of correct ranks / 7.

**Acceptance Target**: Mean depth accuracy ≥ **0.70**

### 3. Noise & Mutation Stress Test

We introduce controlled degradation:

- **Point mutations**: 1%, 5%, 10% divergence
- **Truncations**: 70%, 50% length
- **Chimeras**: Sequence A + Sequence B

**Goal:** Determine at which degradation threshold taxonomy fails.

**Acceptance Criteria:**
- Correct genus-level assignment at 1% mutation
- ≥50% depth accuracy at 5% mutation
- Robust novelty/low-confidence flagging at 10% mutation or chimera

### 4. Vector Consistency (Cloud vs. Edge)

We compare embeddings for the same sequence across environments:

- **Colab (TPU)** vs **Windows (CPU)**
- Cosine similarity **> 0.999** required

This validates the mathematical bridge between cloud and edge inference.

### 5. LanceDB Integrity & Performance

**Latency Test:**

- Measure search latency as the database grows:
  - 1,000 → 5,000 → 10,000 → 100,000 rows

**Disk I/O Profiling:**

- Monitor pendrive read/write during indexing
- Identify storage bottlenecks on the 32GB device

**Acceptance Criteria:**
- Search latency < 250 ms @ 10k rows
- Sustained write speed ≥ 20 MB/s

### 6. False Discovery Rate (FDR)

Feed random DNA sequences (non-biological):

- Expected output: **LOW_CONFIDENCE** or **Out-of-Distribution**
- If AI assigns a lineage to junk DNA, it fails the FDR test.

**Acceptance Target:**
- FDR ≤ **5%**

---

## Hardware Constraints

**Environment:** Windows 11 Laptop

- **RAM:** 16GB (max)
- **Storage:** 32GB pendrive (LanceDB)
- **Model:** NT-500M (768-dim vectors)

**Memory Safety Rule:**

```
Estimated vector memory = num_vectors × embedding_dim × 4 bytes
```

Example:
```
100,000 vectors × 768 × 4 bytes ≈ 307 MB
```

This ensures LanceDB operations remain safe on a 32GB device.

---

## Outputs

The validation suite generates:

- **validation_metrics.json** — raw metrics
- **pipeline_validation_report.md** — report with plots
- **confusion_matrix.png** — taxonomy agreement
- **rarefaction_curve.png** — diversity saturation

---

## Implementation Files

- [tests/pipeline_validator.py](tests/pipeline_validator.py) — end-to-end pipeline validation
- [src/benchmarks/mock_community.py](src/benchmarks/mock_community.py) — mock community utilities
- [tests/validation_report_generator.py](tests/validation_report_generator.py) — report generator
- [tests/conftest.py](tests/conftest.py) — pytest fixtures

---

## Success Criteria Summary

| Test | Target | Status (Run Required) |
|------|--------|------------------------|
| Precision / Recall / F1 | ≥ 0.85 | ☐ pending |
| Mean Depth Accuracy | ≥ 0.70 | ☐ pending |
| Vector Consistency | Cosine ≥ 0.999 | ☐ pending |
| Mutation Robustness | Genus-level at 1% | ☐ pending |
| LanceDB Latency | < 250 ms @ 10k | ☐ pending |
| False Discovery Rate | ≤ 5% | ☐ pending |

---

## Operational Notes (CMLRE Mission)

This validation suite is the **gatekeeper** for discovery claims. No novel taxa should be reported unless:

1. **Mock community validation passes**
2. **Stress tests meet threshold**
3. **FDR remains under control**
4. **Vectors match across cloud and edge**

---

## Next Steps

1. Run `tests/pipeline_validator.py` on the mock community dataset
2. Generate validation report with `tests/validation_report_generator.py`
3. Review metrics against acceptance criteria
4. Proceed with discovery reporting only if all tests pass
