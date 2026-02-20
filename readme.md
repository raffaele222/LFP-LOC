## LFP-LOC

Code associated to the method presented in the paper: "LFP-LOC: an LFP power–based method for the anatomical localization of high-density neural probes".

Estimate a SiNAPS or Neuropixel's probe's anatomical location using the power spectral features present in canonical LFP bands.

### How to use

The script can be run by calling the ```main.py``` program and accepts the following flags [--file or --csv required]:
- --file (Path to the recording, h5/bin/dat/cbin/xml)
- --csv (Path to a CSV containing multiple recordings)
- --ap (AP Coordinates, can be left blank to skip probe placement on atlas)
- --ml (ML Coordinates, can be left blank to skip probe placement on atlas)
- --dv (DV Coordinates, can be left blank to skip probe placement on atlas)
- --start-time (Start time, in seconds, of segment of recording to calculate PSD on, default 0)
- --end-time (End time, in seconds, of segment of recording to calculate PSD on, default 20)
- --custom-format (File format of generated plots, default "png", allowed: "png", "svg", "eps")

A format for the CSV file can be seen in the following:

```
file,ap,ml,dv
C:\path_to_file.h5,1,-4,-4
C:\path_to_bin,,,
C:\path_to_other.xml,2,-3,-5
```

The results for the analysis are saved in the output folder and are structured as following:

output/\<name_of_file>/\<unix_timestamp>/\<time_range_selected>\/<dim_clustering_method>


### How to install

#### With conda:

1. Download the source code
2. CD to the source code you downloaded, and run the following command to create the conda environment:
    
    ```conda env create --file environment.yml```

3. Run the script by running a command similar to the following:

    ```python.exe main.py --file "C:\path\to\rec.h5" --ap 0.5 --ml -3 --dv -4```

#### With pip:

1. Download the source code
2. Run the following command to install all required modules:

``` pip install numpy scikit-learn umap-learn hdbscan matplotlib pandas h5py spikeinterface brainrender ```

3. Run the script with a command similar to the one mentioned above

