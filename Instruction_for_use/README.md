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
