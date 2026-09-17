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


---

## Expanded validation for RNA004 data:

We incorporated our pipeline to RNA004 data, which is more updated than RNA002. Currently we tried on two modifications: m6A and m5C.

The whole workflow is similar to previous part in RNA002. 

### **Software used for RNA004 expanded validation**

| Method | Method type | Version |
|--------|-------------|---------|
| Dorado | Basecaller | v0.6.2 |
| f5c | Eventalign | v1.6 |

---
### **Dataset for RNA004 expanded validation**
| Accession ID | Usage |
|---------|------|
| PRJEB82528 | RNA004 synthetic m6A/m5C training and testing |


**1. Start with basecalling from pod5 raw file:**
```bash
dorado basecaller rna004_130bps_sup@v3.0.1 pod5_dir -x 'cuda:all' > basecall_output_dir/calls.bam
dorado summary basecall_output_dir/calls.bam > basecall_output_dir/calls.summary
samtools fastq basecall_output_dir/calls.bam  > basecall_output_dir/calls.fastq
```

pod5_dir: directory containing pod5 files
basecall_output_dir: output path during basecalling

**2. mapping:**

```bash
mv basecall_output_dir/calls.fastq basecall_output_dir/merge.fastq

# mapping to transcript.fa
minimap2 -ax map-ont -k 14 reference_transcripts.fa -t 25 --secondary=no basecall_output_dir/merge.fastq -o sample_name.sam 

samtools view -@ 30 -F 2048 -F 4 -b sample_name.sam | samtools sort -O BAM -@ 20  -o sample_name.bam
samtools index -@ 16 sample_name.bam

# if your bam file is big, you may split it for parallelly later
# spliting bam files for parallel processing
mkdir split_bam_dir

java -jar picard.jar SplitSamByNumberOfReads --INPUT sample_name.bam --SPLIT_TO_N_FILES 25 --OUTPUT split_bam_dir
for bam in split_bam_dir/*bam
do
{
samtools index $bam
} &
done
```

**3. eventalign**

```bash
mkdir eventalign_output_dir

# making index
pod5 convert to_fast5 pod5_dir/ --output fast5_dir/
f5c index --iop 10 -t 10 -d fast5_dir basecall_output_dir/merge.fastq

# eventalign
f5c eventalign -r basecall_output_dir/merge.fastq -b $file -g reference_transcripts.fa -t 15 --rna --scale-events --samples --signal-index --summary eventalign_output_dir/eventalign_summary.txt --print-read-names > eventalign_output_dir/eventalign.txt
# notice that this eventalign_summary.txt is a new output file, not the same file with calls.summary in step basecalling

#if you need run parallelly for large file:
for file in split_bam_dir/*.bam
do
{
info=(${file//// })
f5c eventalign -r basecall_output_dir/merge.fastq -b $file -g reference_transcripts.fa -t 15 --rna --scale-events --samples --signal-index --summary eventalign_output_dir/eventalign_${info[-1]%%.bam}_summary.txt --print-read-names > eventalign_output_dir/eventalign_${info[-1]%%.bam}_eventalign.txt
} &
done
```

**4. extracting signal from eventalign output and raw data**

```bash
python scripts/extract_signal_from_004_eventalign.py --eventalign eventalign_output_dir/eventalign.txt --reference reference_transcripts.fasta --sam sample_name.sam --fast5_dir fast5_dir/ --output output/output.signal.tsv

# use the same sample_name.sam in step mapping
```

**5. extracting feature**

```bash
python scripts/extract_feature_from_004_signal.py --signal_file output/output.signal.tsv --clip 10 --motif DRACH --output output/output.feature.tsv
```
The --motif argument should take your own input, 'DRACH' is an example for m6A here.

**6. training/testing/prediction**

Then you can follow the same steps as in RNA002 dataset. We provide a pretrained model for simultaneously predicting m6A and m5C, you may try with is by using the cmd below:

```bash
python indep_test_004.py --pretrained models/RNA004_test.pkl --output results/
```

A pretrained RNA004 model are provided as well. You may try with our sample data on the pipeline.



