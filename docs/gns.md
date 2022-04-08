
> Source: https://github.com/geoelements/gns
> 
# Module `graph_network.py`

 The graph network simulator includes $Encoder$ ($\mathcal{X} \to \mathcal{G}$), $Processor$ ($\mathcal{G} \to \mathcal{G}$), and $Decoder$ ($\mathcal{G} \to \mathcal{Y}$). This module includes functions, and classes related to the $\mathcal{X} \to \mathcal{G} \to \mathcal Y$ process. 

 Encoder is simply a multi-layer perceptron (MLP). It accepts particle representation ($X$) as an input. It outputs the properties of a graph ($G$), which are vertice (also called nodes) ($V$) and edges ($E$). Both outputs have a latent dimension of 128. This dimension can be changed by a user. Note that $X$ includes 1) each particle's physical coordinate, 2) particle types (sand, water, ...), and 3) boundaries of the simulation domain. 

 This is a module that includes functions for making a multi-layer perceptron (MLP) object, and classes for encoder, processor, and decoder processes. 
 
 Each particle ($i$) at time ($t$) is represented as $x_{i}^{t_k} = [p_{i}^{t_k}, \dot{p}_{i}^{t_k-C+1}, ..., \dot{p}_{i}^{t_k}, f_i]$ (in the paper) where position, sequence of previous C=5 steps of velocity, static material properties (sand, water, boundary particle, …), respectively. In our code, $x_{i}^{t_k} = [\dot{p}_{i}^{t_k-C+1}, ..., \dot{p}_{i}^{t_k}, f_i]$ whose `shape=(nparticles, 30)` for 2D, where 30 = 10 (5 velocities * 2) + 4 boundaries (top/bottom/left/right) + 16 particles embeddings in 2D case.




## **Function `build_mlp`**



## **Class `InteractionNetwork`**

### Method `__forward__`
### Method `message`
### Method `update`



## **Class `Processor`**


## **Class `Decoder`**


## **Class `EncodeProcessDecode`**






# Module `learned_simulator.py`


## **Class `LearnedSimulator`**

### Method `__init__`

### Method `forward`
### Method `_compute_graph_connectivity`
### Method `_encoder_preprocessor`
### Method `_decoder_postprocessor`
### Method `predict_positions`
### Method `predict_accelerations`
### Method `_inverse_decoder_postprocessor`
### Method `save`
### Method `load`
## **Method `time_diff`**





