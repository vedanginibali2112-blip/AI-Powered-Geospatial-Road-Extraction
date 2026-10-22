

GLOBAL_MAPPER_SCRIPT VERSION=1.00

// Path for the input TIFF file
DEFINE_VAR NAME="INPUT_TIF" VALUE="D:\RoadPlugin\input\T1.tif"

// Path for the output directory
DEFINE_VAR NAME="OUTPUT_DIR" VALUE="D:\RoadPlugin\output"

// Python executable path for the gis_cuda environment
// Path to the Python road detection pipeline script
// Run the pipeline using the input TIFF and output directory
RUN_COMMAND \
COMMAND_LINE="'C:\PATH\TO\gis_cuda\python.exe' 'D:\RoadPlugin\scripts\road_pipeline.py' '%INPUT_TIF%' '%OUTPUT_DIR%'" \
WAIT_FOR_COMPLETE=YES

// Import the generated road shapefile into Global Mapper
IMPORT FILENAME="%OUTPUT_DIR%\Road.shp"

