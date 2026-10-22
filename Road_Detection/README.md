# AI-Powered Geospatial Road Extraction

A deep learning-based geospatial road extraction pipeline using SegFormer and SAM2, designed to identify road regions in georeferenced raster imagery and generate GIS-compatible vector outputs.

## Overview

This project combines semantic segmentation, segmentation refinement, image processing, and geospatial data handling to extract road regions from raster imagery.

The pipeline is designed to integrate with Global Mapper, allowing users to run the Python-based processing workflow and import generated GIS outputs for further inspection.

## Key Technologies

- Python — pipeline execution and processing
- SegFormer — semantic road segmentation
- SAM2 — segmentation refinement
- PyTorch — deep learning inference
- OpenCV — image processing
- Rasterio and GDAL — raster data handling
- GeoPandas and Shapely — geospatial vector processing
- Global Mapper — GIS visualization and workflow integration
- Conda — Python environment management

## Project Structure

```text
AI-Powered-Geospatial-Road-Extraction/
├── config/                 # Configuration files
├── dataset_preparation/    # Dataset preparation resources
├── input/                  # Local input imagery (not committed)
├── logs/                   # Execution logs (not committed)
├── models/                 # Model files and checkpoints
├── output/                 # Generated outputs (not committed)
├── scripts/
│   └── road_pipeline.py    # Main Python pipeline
├── temp/                   # Temporary files (not committed)
├── .gitignore
├── environment.yml
├── README.md
└── Test.gms                # Global Mapper integration script
```

This structure represents the intended organization. Update it if the actual repository differs.

## Requirements

- Windows operating system
- Git
- Anaconda or Miniconda
- Python environment defined in `environment.yml`
- Required trained model checkpoints and configuration files
- Compatible GPU and drivers for GPU inference, if required
- Global Mapper for the integrated GIS workflow

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vedanginibali03-ctrl/AI-Powered-Geospatial-Road-Extraction.git
cd AI-Powered-Geospatial-Road-Extraction
```

Replace the placeholder URL with the actual repository URL.

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate gis_cuda
```

If the environment already exists, activate it instead:

```bash
conda activate gis_cuda
```

### 3. Verify PyTorch and GPU availability

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

GPU availability depends on the installed hardware, drivers, and compatible software versions.

## Model Setup

Place the required trained model checkpoints and configuration files in the locations expected by the Python pipeline.

Before running the project, verify:

- The SegFormer checkpoint path.
- The SAM2 checkpoint path, if required.
- The SAM2 configuration file, if required.
- Any additional configuration files used by the implementation.

Model weights are not necessarily included in this repository. Obtain them only from an authorized source and follow any applicable distribution restrictions.

## Input Data

The intended input is georeferenced raster imagery, such as GeoTIFF files (`.tif` or `.tiff`).

Place authorized input data in the local `input/` directory, or provide the input path directly when running the script.

Supported raster bands, image dimensions, coordinate reference system requirements, and preprocessing requirements depend on the actual implementation.

Do not upload private company imagery, confidential datasets, or restricted geospatial information to GitHub without authorization.

## Running the Pipeline

Activate the Conda environment:

```bash
conda activate gis_cuda
```

Run the Python pipeline using the input raster and output directory.

The following is an example command format. Confirm the argument order against `scripts/road_pipeline.py` before using it:

```bash
python scripts/road_pipeline.py "<INPUT_TIF_PATH>" "<OUTPUT_DIRECTORY>"
```

Replace both placeholders with valid paths on your computer.

The pipeline's actual processing steps, output filenames, and supported options are defined by the source code.

## Global Mapper Integration

The project includes `Test.gms`, intended to connect Global Mapper with the Python processing pipeline.

The script may be configured to:

1. Specify the input raster.
2. Specify the output directory.
3. Launch Python from the Conda environment.
4. Wait for processing to finish.
5. Import the generated GIS output into Global Mapper.

Verify these operations against the current `Test.gms` file. Update local paths and the Python executable path for each computer before running the script.

Global Mapper must be installed and available for the integrated workflow.

## Output

The pipeline is intended to produce GIS-compatible road extraction results.

The exact output filenames and formats must be confirmed from the implementation. If Shapefiles are generated, retain the required associated files, such as `.shp`, `.shx`, `.dbf`, and `.prj`, where applicable.

Inspect the generated results in a GIS application and verify their spatial alignment and geometry quality before using them in downstream work.

## Troubleshooting

### 1. Missing Python Module

Problem: An error such as `ModuleNotFoundError` appears.

Solution:
- Activate the correct Conda environment.
- Verify that the required package is installed.
- Check the dependencies listed in `environment.yml`.

```bash
conda activate gis_cuda
```

### 2. Missing Model Checkpoint

Problem: The pipeline cannot locate a model checkpoint.

Solution:
- Verify that the checkpoint file exists.
- Check the model path configured in the Python script.
- Ensure that you have obtained the required model weights from an authorized source.

### 3. CUDA Unavailable

Problem: PyTorch cannot access the GPU.

Solution:
- Check that compatible GPU hardware and drivers are installed.
- Verify your PyTorch and CUDA configuration.
- Test GPU availability with the following command:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Input Raster Cannot Be Opened

Problem: The pipeline cannot read the input TIFF file.

Solution:
- Verify the input file path.
- Ensure that the file exists and is accessible.
- Check that the raster format and band configuration are supported by the pipeline.

### 5. Global Mapper Cannot Launch Python

Problem: The Global Mapper script fails to start the Python pipeline.

Solution:
- Verify the Python executable path in `Test.gms`.
- Confirm that the `gis_cuda` environment exists.
- Check the input and output paths configured in the script.

### 6. Output Cannot Be Imported

Problem: Global Mapper cannot import the expected GIS output.

Solution:
- Check whether the Python pipeline completed successfully.
- Verify that the expected output file was generated.
- For Shapefiles, ensure that the required associated files are available.

If the issue persists, review the Python console output or execution logs to identify the underlying error.

## Privacy and Data Handling

This repository is configured to exclude selected local input data, generated results, environment files, and model checkpoints through `.gitignore`.

The ignore rules do not remove files that Git already tracks. Review repository contents before pushing changes, and do not publish confidential company data, credentials, or restricted model weights.

## Limitations

Results depend on the input imagery, model weights, image resolution, preprocessing, model configuration, and post-processing parameters.

The pipeline should be evaluated on representative imagery before being used for production GIS work. No specific accuracy level is claimed in this documentation.

## License

A license has not yet been specified. Add an appropriate license only after confirming that you have the rights to distribute the project and its components.

## Acknowledgements

This project uses open-source deep learning and geospatial libraries. Consult the relevant upstream documentation and licenses for SegFormer, SAM2, PyTorch, Rasterio, GDAL, GeoPandas, Shapely, and other dependencies used by the implementation.