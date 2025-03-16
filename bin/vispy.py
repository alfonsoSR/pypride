#!python3

import argparse
from pride.experiment import Experiment
from pride.logger import log


class BuildParser(argparse.ArgumentParser):
    """Argument parser for build script"""

    def __init__(self) -> None:

        super().__init__(
            prog="pride",
            description="Planetary Radio Interferometry Delay Estimator",
        )

        self.add_argument("config", help="Path to configuration file")
        self.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Print verbose output",
        )

        return None


def main():

    args = BuildParser().parse_args()

    if args.verbose:
        log.setLevel("DEBUG")

    experiment = Experiment(args.config)
    with experiment.spice_kernels():

        for baseline in experiment.baselines:

            # Update baseline with data from observations
            baseline.update_with_observations()

            # Update station coordinates with geophysical displacements
            baseline.update_station_with_geophysical_displacements()

            for observation in baseline.observations:

                # Calculate spherical coordinates of the source
                observation.update_with_source_coordinates()

                # Calculate delays for the observation
                observation.calculate_delays()

    experiment.save_output()
