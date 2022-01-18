import os
import sys
import argparse

import torch

from gns.train import get_ds

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group("gloo",
                                         rank=rank,
                                         world_size=world_size,
                                         backend="nccl"
                                        )

def cleanup():
    torch.distributed.destroy_process_group()

def make_model():
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
    return simulator

def train(rank, world_size):
    print(f"Running train on rank {rank} of world_size {world_size}.")
    setup(rank, world_size)

    model = make_model().to(rank)
    ddp_model = torch.nn.distirbuted.DistributedDataParallel(model, device_ids=[rank])
 
    optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
    step = 0

    try:
        for features, labels in get_data_loader():
            features['position'] = torch.tensor(features['position']).to(device)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device)
            labels = torch.tensor(labels).to(device)

            # Sample the noise to add to the inputs to the model during training.
            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(features['position'], noise_std_last_step=FLAGS.noise_std).to(device)
            non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
            sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

            # Get the predictions and target accelerations.
            pred_acc, target_acc = simulator.predict_accelerations(
                next_positions=labels.to(device),
                position_sequence_noise=sampled_noise.to(device),
                position_sequence=features['position'].to(device),
                nparticles_per_example=features['n_particles_per_example'].to(device),
                particle_types=features['particle_type'].to(device))

            # Calculate the loss and mask out loss on kinematic particles
            loss = (pred_acc - target_acc) ** 2
            loss = loss.sum(dim=-1)
            num_non_kinematic = non_kinematic_mask.sum()
            loss = torch.where(non_kinematic_mask.bool(),
                                loss, torch.zeros_like(loss))
            loss = loss.sum() / num_non_kinematic

            # Computes the gradient of loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step/FLAGS.lr_decay_steps))
            for param in optimizer.param_groups:
                param['lr'] = lr_new

            if rank == 0:
                print(f"Training step: {step}/{FLAGS.ntraining_steps}. Loss: {loss}.")

            # Save model state
            # if step % FLAGS.nsave_steps == 0:
            #     simulator.save(model_path + 'model-'+str(step)+'.pt')
            #     train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
            #     torch.save(optimizer.state_dict(), f"{model_path}train_state-{step}.pt")

            # Complete training
            if (step > FLAGS.ntraining_steps):
                break

            step += 1

    except KeyboardInterrupt:
        pass
    
    cleanup()

    # simulator.save(model_path + 'model-'+str(step)+'.pt')
    # train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
    # torch.save(train_state, f"{model_path}train_state-{step}.pt")


def spawn_train(train_fxn, world_size):
    torch.multiprocessing(train_fxn,
                          args=(world_size,),
                          nproces=world_size,
                          join=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Distributed Training.')
    parser.add_argument("--nprocs", "-n", type=int,
                        help='Number of processes to spawn.')
    

    args = parser.parse_args()

    # Sensitive to env variable USE_DISTRIBUTED in {0,1}
    # Linux and Windows = 1 by default
    # MacOS = 0 by default
    print(f"torch.distributed={torch.distributed.is_available()}")

    spawn_train(train, args.nprocs)

