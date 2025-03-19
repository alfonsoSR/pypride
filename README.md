# PRIDE: Planetary Radio-Interferometry Delay Estimator

## Installation

```
git clone https://github.com/alfonsoSR/pypride.git
cd pypride
pip3 install .
```

## Example

You can find an example application under `examples`. You should be able to run the program `main.py` to calculate the delays for the GR035 experiment. Binary output files will be saved to `examples/output-gr035`. You can use the script `examine_results.py` to plot the delays in these files.

The first time you run the program it will take a little bit because it has to download kernels and data files. If you already have the data files and kernels available, the execution should take about 3 minutes.

If you want to turn off the DEBUG logs, you can set the log level to `INFO` in `main.py`
