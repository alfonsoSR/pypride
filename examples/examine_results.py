from pride import io
from pathlib import Path
from nastro import plots as ng
import numpy as np

# Code of the station to be examined and output directory
station = "Cd"
output_directory = Path("output-gr035").resolve()

if __name__ == "__main__":

    # Read data from binary file of station
    source = output_directory / f"GR035_{station}.del"
    _, data = io.DelFile(source).read()

    # Plot setup
    setup = ng.PlotSetup(
        figsize=(6.0, 6.0),
        xlabel="Hours past initial epoch",
        ylabel="Delay [s]",
        dir=".",
        name="validation.png",
        save=False,
        show=True,
    )

    # Plot data
    with ng.SingleAxis(setup) as fig:

        mjd_00 = data[0].mjd_ref + (data[0].mjd2[0] / 86400)

        for idx, scan in enumerate(data):

            t = ((scan.mjd_ref + (scan.mjd2 / 86400)) - mjd_00) * 24.0
            fig.add_line(t, scan.delays, ".")
