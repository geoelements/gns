
import torch

from torch.gns.train import 

if __name__ == "__main__":

    # Sensitive to env variable USE_DISTRIBUTED={0,1}
    # Linux and Windows = 1 by default
    # MacOS = 0 by default
    print(f"torch.distributed={torch.distributed.is_available()}")

    # Initialize group, blocks until all processes join.
    torch.distributed.init_process_group()

    # Set CUDA_VISIBLE_DEVICES=0,1 
    # torch.cuda.set_device(args.local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model
                                                      device_ids=[args.local_rank])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_datset,
        num_replicas=args.backend.size(),
        rank=args.backen.rank()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size * args.batches_per_allreduce,
        sampler=train_sampler,
        **kwargs
    )



def train(
        simulator: learned_simulator.LearnedSimulator):
  """Train the model.

  Args:
    simulator: Get LearnedSimulator.
  """
  # If model_path does not exist create new directory and begin training.
  model_path = FLAGS.model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
    step = 0

  # If model_path does exist and model_file and train_state_file exist continue training.
  elif os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
      # load model
      simulator.load(model_path + FLAGS.model_file)

      # load train state
      train_state = torch.load(model_path + FLAGS.train_state_file)
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      # set global train state
      step = train_state["global_train_state"].pop("step")
 
  else:
    msg = f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found."
    raise FileNotFoundError(msg) 

  simulator.train()
  simulator.to(device)

  ds = prepare_input_data(FLAGS.data_path,
                          batch_size=FLAGS.batch_size)

  print(f"device = {device}")
  try:
    for features, labels in ds:
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

      print('Training step: {}/{}. Loss: {}.'.format(step,
                                                     FLAGS.ntraining_steps,
                                                     loss))
      # Save model state
      if step % FLAGS.nsave_steps == 0:
        simulator.save(model_path + 'model-'+str(step)+'.pt')
        train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
        torch.save(optimizer.state_dict(), f"{model_path}train_state-{step}.pt")

      # Complete training
      if (step > FLAGS.ntraining_steps):
        break

      step += 1

  except KeyboardInterrupt:
    pass

  simulator.save(model_path + 'model-'+str(step)+'.pt')
  train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
  torch.save(train_state, f"{model_path}train_state-{step}.pt")

