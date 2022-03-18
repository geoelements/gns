import argparse
from msilib.schema import Directory
import pathlib
import glob
import re

import h5py
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert hdf5 trajectories to npz.')
    parser.add_argument('--path', nargs="+", help="Path(s) to hdf5 files to consume.")
    parser.add_argument('--ndim', default=2, help="Dimension of input data, default is 2 (i.e., 2D).")
    parser.add_argument('--output', help="Name of the output file.")
    args = parser.parse_args()

    directories = [pathlib.Path(path) for path in args.path]

    for directory in directories:
        if not directory.exits():
            raise FileExistsError(f"The path {directory} does not exist.")
    print(f"Number of trajectories: {len(directories)}")

    # setup up variables to calculate on-line mean and standard deviation
    # for velocity and acceleration.
    if int(args.ndim) == 2:
        running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_diff = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    else:
        raise NotImplementedError

    trajectories = {}
    for nth_trajectory, directory in enumerate(directories):
        fnames = glob.glob("*.h5")
        get_fnumber = re.compile(".*(\d+).hdf5")
        fnumber_and_fname = [(int(get_fnumber.findall(fname)[0]), fname) for fname in fnames]
        fnumber_and_fname_sorted = sorted(fnumber_and_fname, key=lambda row: row[0])

        # get size of trajectory
        with h5py.File(fnames[0], "r") as f:
            nparticles, ndim = f["table"]["coord_x"].shape
        nsteps = len(fnames)

        # allocate memory for trajectory
        # assume number of particles does not change along the rollout.
        positions = np.empty((nsteps, nparticles, ndim), dtype=float)
        print(f"Size of trajectory {nth_trajectory} ({directory.parent}): {positions.shape}")

        # open each file and copy data to positions tensor.
        for nth_step, fname in enumerate(fnames):
            with h5py.File(fname, "r") as f:
                positions[nth_step, :, 0] = f["table"]["coord_x"][:]
                positions[nth_step, :, 1] = f["table"]["coord_y"][:]

                # update variables for on-line mean and standard deviation calculation.
                for key in running_sum:
                    running_sum[key] += np.sum(f["table"][key])
                    running_diff[key] += np.sum(f["table"][key] - np.mean(f["table"][key]))
                    running_count[key] += np.size(f["table"][key])

        trajectories[str(directory.parent)] = (positions, np.full(positions.shape[1], 6, dtype=int))

    # compute online mean and standard deviation.
    print("Statistis across all trajectories:")
    for key in running_sum:
        mean = running_sum[key] / running_count[key]
        std = np.sqrt(running_diff[key]**2 / running_count[key])
        print(f"  {key}: mean={mean:.4f}, std={std:.4f}")

    np.savez_compressed(args.output, **trajectories)
    print(f"Output written to: {args.output}")