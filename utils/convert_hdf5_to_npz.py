import argparse
import pathlib
import glob
import re

import h5py
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert hdf5 trajectories to npz.')
    parser.add_argument('--path', nargs="+", help="Path(s) to hdf5 files to consume.")
    parser.add_argument('--ndim', default=2, help="Dimension of input data, default is 2 (i.e., 2D).")
    parser.add_argument('--dt', default=2E-4, help="Time step between position states.")
    parser.add_argument('--output', help="Name of the output file.")
    args = parser.parse_args()

    directories = [pathlib.Path(path) for path in args.path]

    for directory in directories:
        if not directory.exists():
            raise FileExistsError(f"The path {directory} does not exist.")
    print(f"Number of trajectories: {len(directories)}")

    # setup up variables to calculate on-line mean and standard deviation
    # for velocity and acceleration.
    ndim = int(args.ndim)
    if ndim == 2:
        running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    else:
        raise NotImplementedError

    trajectories = {}
    for nth_trajectory, directory in enumerate(directories):
        fnames = glob.glob(f"{str(directory)}/*.h5")
        get_fnumber = re.compile(".*\D(\d+).h5")
        fnumber_and_fname = [(int(get_fnumber.findall(fname)[0]), fname) for fname in fnames]
        fnumber_and_fname_sorted = sorted(fnumber_and_fname, key=lambda row: row[0])
        
        # get size of trajectory
        with h5py.File(fnames[0], "r") as f:
            (nparticles,) = f["table"]["coord_x"].shape
        nsteps = len(fnames)

        # allocate memory for trajectory
        # assume number of particles does not change along the rollout.
        positions = np.empty((nsteps, nparticles, ndim), dtype=float)
        print(f"Size of trajectory {nth_trajectory} ({directory}): {positions.shape}")

        # open each file and copy data to positions tensor.
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                positions[nth_step, :, 0] = f["table"]["coord_x"][:]
                positions[nth_step, :, 1] = f["table"]["coord_y"][:]
        
        dt = float(args.dt)
        # compute velocities using finite difference
        # assume velocities before zero are equal to zero
        velocities = np.empty_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1])/dt
        velocities[0] = 0

        # compute accelerations finite difference
        # assume accelerations before zero are equal to zero
        accelerations = np.empty_like(velocities)
        accelerations[1:] = (velocities[1:] - velocities[:-1])/dt
        accelerations[0] = 0

        # update variables for on-line mean and standard deviation calculation.
        for key in running_sum:
            if key == "velocity_x":
                data = velocities[:,:,0]
            elif key == "velocity_y":
                data = velocities[:,:,1]
            elif key == "acceleration_x":
                data = accelerations[:,:,0]
            elif key == "acceleration_y":
                data = accelerations[:,:,1]
            else:
                raise KeyError

            running_sum[key] += np.sum(data)
            running_sumsq[key] += np.sum(data**2)
            running_count[key] += np.size(data)

        trajectories[str(directory)] = (positions, np.full(positions.shape[1], 6, dtype=int))

    # compute online mean and standard deviation.
    print("Statistis across all trajectories:")
    for key in running_sum:
        mean = running_sum[key] / running_count[key]
        std = np.sqrt((running_sumsq[key] - running_sum[key]**2/running_count[key]) / (running_count[key] - 1))
        print(f"  {key}: mean={mean:.4f}, std={std:.4f}")

    np.savez_compressed(args.output, **trajectories)
    print(f"Output written to: {args.output}")
