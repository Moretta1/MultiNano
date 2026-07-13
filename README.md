# **MultiNano**
MultiNano is a deep learning framework for **simultaneous prediction of seven RNA modifications** from **nanopore direct RNA sequencing (DRS)** data. It aims to simultaneously detect and interpret seven common RNA modifications: hm5C, I, m1A, m5C, m6A, m7G, and Ψ.

---

## **Features**
- **Multi-label prediction**: Predict multiple RNA modifications simultaneously.
- **Comprehensive analysis**: Evaluate both binary and multi-label classification metrics.
- **High performance**: Demonstrated competitive results compared to baseline methods (you may try the baseline methods in file models.py).
- **Flexible dataset compatibility**: Supports both synthetic datasets for model development and native RNA datasets for independent evaluation, supports RNA002 data. 

---

## **Installation**

To use MultiNano, follow these steps:

Clone this repository:
   ```bash
   git clone git@github.com:Moretta1/MultiNano.git
   cd MultiNano
   ```
or you can simply download the .zip file for further usage

---
## **Software used in this study**

The following software packages were used for data preprocessing, model training, benchmarking, and evaluation in this study.

| Method | Method type | Version |
|--------|-------------|---------|
| MultiNano | Predictor | 1.0 |
| SingleMod | Predictor | latest available release |
| m6Anet | Predictor | 2.1.0 |
| MINES | Predictor | latest available release |
| DENA | Predictor | latest available release |
| ORCA | Predictor | latest available release |
| Nanom6A | Predictor | 2.0 |
| Tombo | Annotation and resquiggling | 1.5.1 |
| Nanopolish | Eventalign | 0.11.3 |
| Guppy | Basecaller | 6.5.7 |

---
## Datasets

MultiNano was developed and evaluated using several publicly available nanopore direct RNA sequencing datasets.

| Accession ID | Usage |
|---------|------|
| SRP166020 | Model training and internal validation |
| GSE227087 | Independent read-level testing |
| PRJEB40872 | Site-level evaluation |
| GSE210563 | GLORI ground truth |
| PRJEB81662 | Native rRNA evaluation |
| PRJEB55684 | Native tRNA evaluation |

---
## **Usage**

To predict RNA modifications:
Pre-processing of the raw fast5 files: we use Guppy v6.5.7 for basecalling first, then Tombo v1.5.1 for resquiggling process.

**0.  check whether your fast5 file is multi-fast5 or single fast5:**
  ```bash
  du -sh xxx.fast5
  ```
For a multi-fast5 file, usually it is of size 200-300M; while for a single-fast5 file, usually it is of several hundard of k.

Split those multi-fast5 into single-fast5 file, you can achieve it via:
  ```bash
  multi_to_single_fast5 -i multi-fast5/file -s single-fast5/files -t 40 --recursive
  ```

**1. Guppy basecalling:**

```bash
 guppy_basecaller -i single-fast5/files -s guppy_output/ --config ../data/rna_r9.4.1_70bps_fast.cfg -r --num_callers 4 --cpu_threads_per_caller 2 -x 'cuda:0'
```

here you should replace the --config parameter input with your actual config path, and you can merge all the fastq files to a combined one:
```bash
cat *.fastq > all.guppy.fastq
```

**2. Tombo resquiggle:**

first annotate_raw_with_fastqs:
```bash
tombo preprocess annotate_raw_with_fastqs --fast5-basedir single-fast5/files --fastq-filenames guppy_output/all.guppy.fastq --sequencing-summary-filenames guppy_output/sequencing_summary.txt --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
```

then resquiggle with Tombo:
```
tombo resquiggle --overwrite --basecall-group Basecall_1D_001 demo/fast5_dir  demo/reference_transcripts.fasta --processes 40 --fit-global-scale --include-event-stdev
```
  
**3.	Extract signals from fast5 files:**

We first use minimap2(v2.24-r1122) to map basecalled sequences to **reference transcripts**: 

```bash
minimap2 -ax map-ont reference_transcripts.fasta guppy_output/all.guppy.fastq > guppy_output/output.sam
```
Then we can extract signal files from FAST5 files:

```bash
python scripts/extract_signal_from_fast5.py -p 40 --fast5 single-fast5/files --reference reference_transcripts.fasta --sam guppy_output/output.sam --output output/output.signal.tsv --clip 10
```

**4.	Extract features from signals:**
```bash
python scripts/extract_feature_from_signal.py  --signal_file output/output.signal.tsv --clip 10 --output output/output.feature.tsv --motif DRACH
```
The --motif argument should take your own input, DRACH is an example for m6A here.

**5. Then you can use your features for training/testing/prediction in MultiNano pipeline:**

Using prediction of real HEK293T dataset, m6A modification as an example here:

```bash 
python scripts/predict.py --type m6A --pretrained_model models/bs_512_lr_0.00001/epoch4.pkl --feature_file output/output.feature.tsv --predict_result output/predict_output.tsv --bs 512
```
We have contained the usage of ELIGOS dataset with 7 modifications and IVET rice dataset with 3 modifications in the file 'train.py' and 'ivet-testing.py' respectively.

You may also change the input arguement '--type all' to get the prediction score of all the 7 modification.

The pretrained models released in this repository were trained on the IVT synthetic dataset (SRP166020).

We have attached the code of three baseline methods in file 'models.py', you may switch to the model that you are interested in by changing the class name in train/testing file.

### Training Notes

The model was implemented in **PyTorch** and optimized using the **Adam** optimizer. 
The training hyperparameters used in this study are summarized below.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Initial learning rate | `1e-5` |
| Batch size | `256` |
| Learning rate decay | `0.0025` |
| Loss weighting | Uncertainty weighting |
| Hard example mining | Online Hard Example Mining (OHEM) |
| Early stopping | Validation convergence |



