# RenderDoc CSV to USD

Small utility functions to convert CSV files from RenderDoc to USD.

This script depends on explicit mesh data layouts, but can easily be modified for other formats.

Usage (Windows):\
`python C:\path\to\script\renderdoc_csv_to_usd.py file.csv`

It will create a USD file with the same name next to it.

To process all CSV files in a folder (Windows):\
`for %f in (*.csv) do python C:\path\to\script\renderdoc_csv_to_usd.py "%f"`
