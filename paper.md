---
title: 'GNS: A generalizable Graph Neural Network-based simulator for particulate and fluid modeling'
tags:
  - Python
  - machine learning
  - simulation
authors:
  - name: Krishna Kumar
    orcid: 0000-0003-2144-5562
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Joseph Vantassel
    orcid: 0000-0002-1601-3354
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: "2,3"

affiliations:
 - name: Assistant Professor, University of Texas at Austin, Texas, USA
   index: 1
 - name: Assistant Professor, Virginia Tech, Virginia, USA
   index: 2
 - name: Texas Advanced Computing Center, University of Texas at Austin, Texas, USA
   index: 3
date: 16 October 2022
bibliography: references.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
---

# Summary

Graph Network-based Simulator (GNS) is a framework for developing generalizable, efficient, and accurate machine learning (ML)-based surrogate models for particulate and fluid systems using Graph Neural Networks (GNNs).  GNNs are the state-of-the-art geometric deep learning (GDL) that operates on graphs to represent rich relational information [@scarselli2008graph], which maps an input graph to an output graph with the same structure but potentially different node, edge, and global feature attributes.  The graph network in GNS spans the physical domain with nodes representing an individual or a collection of particles, and the edges connecting the vertices representing the local interaction between particles or clusters of particles.  The GNS computes the system dynamics via learned message passing.  \autoref{fig:gns} shows an overview of how GNS learns to simulate n-body dynamics.  The GNS has three components: (a) Encoder, which embeds particle information to a latent graph, the edges represent learned functions; (b) Processor, which allows data propagation and computes the nodal interactions across steps; and (c) Decoder, which extracts the relevant dynamics (e.g., particle acceleration) from the graph.  The GNS learns the dynamics, such as momentum and energy exchange, through a form of messages passing [@gilmer2017neural], where latent information propagates between nodes via the graph edges.  The GNS edge messages  ($e^\prime_k \leftarrow \phi^e(e_k, v_{r_k}, v_{s_k}, u)$) are a learned linear combination of the interaction forces.  The edge messages are aggregated at every node exploiting the principle of superposition $\bar{e_i^\prime} \leftarrow \sum_{r_k = i} e_i^\prime$.  The node then encodes the connected edge features and its local features using a neural network: $v_i^\prime \leftarrow \phi^v (\bar{e_i}, v_i, u)$.  

![An overview of the graph network simulator (GNS).\label{fig:gns}](figs/gnn.png)

The GNS implementation uses semi-implicit Euler integration to update the state of the particles based on the nodes predicted accelerations.  We introduce physics-inspired simple inductive biases, such as an inertial frame that allows learning algorithms to prioritize one solution over another, instead of learning to predict the inertial motion, the neural network learns to trivially predict a correction to the inertial trajectory, reducing learning time.  We developed an open-source, PyTorch-based GNS that predicts the dynamics of fluid and particulate systems [@Kumar_Graph_Network_Simulator_2022].  GNS trained on trajectory data is generalizable to predict particle kinematics in complex boundary conditions not seen during training.  \autoref{fig:gns-mpm} shows the GNS prediction of granular flow around complex obstacles trained on 20 million steps with 40 trajectories on NVIDIA A100 GPUs.  The trained model accurately predicts within 5\% error of its associated material point method (MPM) simulation.  The predictions are 5,000x faster than traditional MPM simulations (2.5 hours for MPM simulations versus 20 s for GNS simulation of granular flow) and are widely used for solving optimization, control and inverse-type problems.  In addition to surrogate modeling, GNS trained on flow problems is also used as an oracle to predict the dynamics of flows to identify critical regions of interest for in situ rendering and visualization [@kumar2022insitu].  The GNS code is distributed under the open-source MIT license and is available on [GitHub Geoelements GNS](https://github.com/geoelements/gns).

![GNS prediction of granular flow on ramps, compared against MPM simulation.\label{fig:gns-mpm}](figs/gns-mpm.png){ width=80% }

# Statement of need

Traditional numerical methods for solving differential equations are invaluable in scientific and engineering disciplines.  However, such simulators are computationally expensive and intractable for solving large-scale and complex inverse problems, multiphysics, and multi-scale mechanics.  Surrogate models trade off generality for accuracy in a narrow setting.  Recent growth in data availability has spurred data-driven machine learning (ML) models that train directly from observed data [@prume2022model].  ML models require significant training data to cover the large state space and complex dynamics.  Instead of ignoring the vast amount of structured prior knowledge (physics), we can exploit such knowledge to construct physics-informed ML algorithms with limited training data.  GNS uses static and inertial priors to learn the interactions between particles directly on graphs and can generalize with limited training data [@wu2020comprehensive;@velivckovic2017graph].  Graph-based GNS offer powerful data representations of real-world applications, including particulate systems, material sciences, drug discovery, astrophysics, and engineering [@sanchez2020learning;@battaglia2018relational].

# State of the art
Numerical methods, such as particle-based approaches or continuum strategies like Material Point Method [@soga2016] and the Finite Element Method, serve as valuable tools for modeling a wide array of real-world engineering systems. Despite their versatility, these traditional numerical methods often prove computationally intensive, restricting them to a handful of simulations. With the growth of material sciences and the escalating complexity of engineering challenges, there is a pressing need to navigate expansive parametric spaces and solve complex optimization and inverse analysis. However, the computational bottleneck inherent in traditional methods thwarts our ability to achieve innovative data-driven discoveries. A surrogate model presents a solution to this hurdle. However, most current neural network-based surrogates operate as black box algorithms, lacking physics and underperforming when extrapolation beyond training regions is needed. Consequently, there is a need to develop generalizable surrogate models based on physics to bridge this gap effectively.

@sanchez2020learning developed a reference GNS implementation based on TensorFlow v1 [@tensorflow2015whitepaper].  Although the reference implementation runs both on CPU and GPU, it doesn't achieve multi-GPU scaling.  Furthermore, the dependence on TensorFlow v1 limits its ability to leverage features such as eager execution in TF v2.  We develop a scalable and modular GNS using PyTorch using the Distributed Data Parallel model to run on multi-GPU systems.

# Key features 

The Graph Network Simulator (GNS) uses PyTorch and PyTorch Geometric for constructing graphs and learned message passing. GNS is highly-scalable to 100,000 vertices and more than a million edges. The PyTorch GNS supports the following features:

- CPU and GPU training
- Parallel training on multi-GPUs
- Multi-material interactions
- Complex boundary conditions
- Checkpoint restart
- VTK results
- Animation postprocessing

# GNS training and prediction

GNS models are trained on 1000s of particle trajectories from MPM (for sands) and Smooth Particle Hydrodynamics (for water) for 20 million steps. GNS predicts the rollout trajectories of particles, based on its training of MPM particle simulations. We employ Taichi MPM [@hu2018mlsmpmcpic] to compute the particle trajectories. The input to GNS includes the velocity context for five timesteps. GNS computes the acceleration between the five timesteps using the timestep $\delta t$. GNS then rolls out the next states $\boldsymbol{X}_{i+5} \rightarrow, \ \ldots, \rightarrow \boldsymbol{X}_k$, where $\boldsymbol{X}$ is the set of particle positions. We use the `.npz` format to store the training data, which includes a list of tuples of arbitrary length where each tuple corresponds to a differenet training trajectory and is of the form `(position, particle_type)`. The position is a 3-D tensor of shape `(n_time_steps, n_particles, n_dimensions)` and particle_type is a 1-D tensor of shape `(n_particles)`.

# Parallelization and scaling

The GNS is parallelized to run across multiple GPUs using the PyTorch Distributed Data Parallel (DDP) model.  The DDP model spawns as many GNS models as the number of GPUs, distributing the dataset across all GPU nodes.  Consider, our training dataset with 20 simulations, each with 206 time steps of positional data $x_i$, which yields $(206 - 6) \times 20 = 4000$ training trajectories.  We subtract six position from the GNS training dataset as we utilize five previous velocities, computed from six positions, to predict the next position.  The 4000 training tajectories are subsequently distributed equally to the four GPUs (1000 training trajectories/GPU).  Assuming a batch size of 2, each GPU handles 500 trajectories in a batch.  The loss from the training trajectories are computed as difference between accelerations of GNS prediction and actual trajectories. 

$$f(\theta) = \frac{1}{n}\sum_{i=1}^n ((\ddot{x}_t^i)_{GNS} - (\ddot{x}_t^i)_{actual})\,,$$

where $n$ is the number of particles (nodes) and $\theta$ is the learnable parameter in the GNS. In DDP, the gradient $\nabla (f(\theta))$ is computed as the average gradient across all GPUs as shown in \autoref{fig:gns-ddp}.

![Distributed data parallelization in GNS.\label{fig:gns-ddp}](figs/gns-ddp.png){ width=80% }


We tested the strong scaling of the GNS code on a single node of Lonestar 6 at the Texas Advanced Computing Center equipped with three NVIDIA A100 GPUs.  We evaluated strong scaling for the [WaterDropSample](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published//PRJ-3702/WaterDropSample) dataset for 6000 training steps using the recommended `nccl` DDP backend.  \autoref{fig:gns-scaling} shows linear strong scaling performance.

![GNS strong-scaling on up to three NVIDIA A100 GPUs.\label{fig:gns-scaling}](figs/gns-scaling.png)

# Acknowledgements

We acknowledge the support of National Science Foundation NSF OAC: 2103937.

# References
