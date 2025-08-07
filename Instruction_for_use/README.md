# Step by Step Instruction

## Data
The data is provided in `/sample_data`, which include the patient chest CT scan. AirwayNet processes these CT scans as input to generate binary airway models and hierarchical anatomical labels.

## AirwayNet Process
The dicated pipeline wrapper in `airwayatlas_pipeline.py`, simply execute the following command:

```
python airwayatlas_pipeline.py
```

You will see similar log messages as follows:

```
(INFO) AirwayNet: Binary Airway initialized.
(INFO) AirwayNet: Multi-class Airway Anatomy initialized. 
(INFO) AirwayNet: Lobe extraction complete.
(INFO) AirwayNet: Airway modeling complete.
(INFO) AirwayNet: Skeleton computation complete.
(INFO) AirwayNet: Saving complete.
(INFO) AirwayNet: Finished tree-parsing.
(INFO) AirwayNet: Finished feature extraction.
(INFO) AirwayNet: Finished inference dataset building.
(INFO) AirwayNet: Finished prediction.
```

## Data input and output structure
Data input structure:

```
project_root/
├── sample_data/              # Input directory containing patient chest CT scans
│   ├── patient_01/
│   │   ├── image.nii.gz      # CT scan file
```

After the AirwayNet processing is complete, the main results are, by default, saved in the same directory as the input data:

```
project_root/
├── sample_data/              # Input directory containing patient chest CT scans
│   ├── patient_01/
│   │   ├── image.nii.gz                             # CT scan file
│   │   ├── airway_bin.nii.gz                        # binary airway structure
│   │   ├── patient_01_pred_lob.nii.gz               # lobar airway anatomy
│   │   ├── patient_01_pred_seg.nii.gz               # segmental airway anatomy
│   │   ├── patient_01_pred_sub.nii.gz               # subsegmental airway anatomy
```

Meanwhile, the efficient Branching Pattern Analysis can be found in ```branchingpattern/airwaybranchpattern_pipeline.py```. The morphological airway signatures can be found in ```features/airway_morph_features.py```. 
These are described in detail in the `Method section` with accompanying pseudocode.

## Expected run time
AirwayNet was deployed and tested on a system with the following hardware configuration: a 12th Gen Intel® Core™ i9-12900KF CPU, 64 GB of system memory, and an NVIDIA RTX 3090 GPU with 24 GB of VRAM. 

The pipeline processes relatively large 3D CT scans (with a typical volume size of approximately 700×512×512) as input and requires about 5–7 minutes per case to complete the entire process.
