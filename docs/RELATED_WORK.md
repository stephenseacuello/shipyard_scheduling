# Related Work

This document surveys the relevant literature for health-aware shipyard scheduling using reinforcement learning and graph neural networks.

## 1. Shipyard Scheduling and Block Assembly

### 1.1 Traditional Approaches

**Lee, K., & Kim, H. (2019).** "An integrated scheduling system for block assembly in shipbuilding." *Computers & Industrial Engineering*, 127, 1-15.
- Mixed-integer programming formulation for block assembly scheduling
- Considers spatial constraints and precedence relationships
- Limitation: Does not consider equipment health or stochastic events

**Song, Y., Lee, K., & Woo, J. H. (2021).** "Spatial scheduling for mega-block assembly in shipbuilding." *Journal of Manufacturing Systems*, 58, 241-253.
- Multi-objective optimization for spatial allocation
- Considers crane interference and transport paths
- Limitation: Static scheduling without real-time adaptation

**Cho, K. K., Sun, J. G., & Oh, J. S. (1999).** "An automated welding operation planning system for block assembly in shipbuilding." *International Journal of Production Economics*, 60, 203-209.
- Early work on automated planning for shipyard operations
- Focus on welding sequence optimization

### 1.2 Metaheuristics for Shipyard Scheduling

**Zheng, J., et al. (2020).** "A genetic algorithm for ship block assembly sequence planning." *Ocean Engineering*, 212, 107628.
- Genetic algorithm with custom operators for block sequencing
- Considers spatial and temporal constraints
- Shows 15% improvement over manual planning

**Park, H., & Lee, K. (2014).** "Block assembly planning using case-based reasoning." *Expert Systems with Applications*, 41(12), 5565-5574.
- Knowledge-based approach leveraging historical data
- Learns from past scheduling decisions

## 2. Reinforcement Learning for Scheduling

### 2.1 RL for Job-Shop Scheduling

**Zhang, C., Song, W., Cao, Z., et al. (2020).** "Learning to dispatch for job shop scheduling via deep reinforcement learning." *Advances in Neural Information Processing Systems*, 33.
- GNN-based policy for job-shop scheduling
- Shows generalization across problem sizes
- **Key insight**: Graph representation captures problem structure effectively

**Park, J., Chun, J., Kim, S. H., et al. (2021).** "Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning." *International Journal of Production Research*.
- Combines heterogeneous GNN with PPO
- Demonstrates transfer learning capabilities

### 2.2 RL for Vehicle Routing

**Nazari, M., Oroojlooy, A., Snyder, L. V., & Takác, M. (2018).** "Reinforcement learning for solving the vehicle routing problem." *Advances in Neural Information Processing Systems*, 31.
- Attention-based model for VRP
- End-to-end learning without hand-crafted features

**Kool, W., van Hoof, H., & Welling, M. (2019).** "Attention, learn to solve routing problems!" *International Conference on Learning Representations*.
- Transformer architecture for combinatorial optimization
- State-of-the-art results on TSP and VRP benchmarks

### 2.3 RL for Production Scheduling

**Waschneck, B., et al. (2018).** "Deep reinforcement learning for semiconductor production scheduling." *IEEE International Conference on Automation Science and Engineering*.
- DQN for semiconductor fab scheduling
- Handles complex production constraints

**Liu, C. L., Chang, C. C., & Tseng, C. J. (2020).** "Actor-critic deep reinforcement learning for solving job shop scheduling problems." *IEEE Access*, 8, 71752-71762.
- A2C algorithm for job-shop problems
- Comparison with dispatching rules

## 3. Graph Neural Networks for Combinatorial Optimization

### 3.1 GNN Architectures

**Kipf, T. N., & Welling, M. (2017).** "Semi-supervised classification with graph convolutional networks." *International Conference on Learning Representations*.
- Foundational GCN architecture
- Spectral graph convolutions

**Veličković, P., et al. (2018).** "Graph attention networks." *International Conference on Learning Representations*.
- Attention mechanism for graphs (GAT)
- Adaptive neighbor weighting
- **Used in our architecture**: Multi-head GAT for message passing

**Schlichtkrull, M., et al. (2018).** "Modeling relational data with graph convolutional networks." *European Semantic Web Conference*.
- Relational GCN for heterogeneous graphs
- Foundation for typed edge handling

### 3.2 GNN for Optimization

**Cappart, Q., Chételat, D., Khalil, E., et al. (2021).** "Combinatorial optimization and reasoning with graph neural networks." *Journal of Machine Learning Research*, 24(130), 1-61.
- Comprehensive survey of GNN for CO
- Taxonomy of problem representations
- **Key reference** for our approach

**Almasan, P., et al. (2022).** "Deep reinforcement learning meets graph neural networks: Exploring a routing optimization use case." *Computer Communications*, 196, 184-194.
- GNN-RL for network routing
- Similar heterogeneous graph structure

### 3.3 Heterogeneous GNNs

**Wang, X., et al. (2019).** "Heterogeneous graph attention network." *The World Wide Web Conference*.
- HAN for heterogeneous information networks
- Type-specific transformations

**Hu, Z., et al. (2020).** "Heterogeneous graph transformer." *The Web Conference*.
- Transformer-style attention for heterogeneous graphs
- Scalable to large graphs

## 4. Equipment Health Management and PHM

### 4.1 Degradation Modeling

**Si, X. S., Wang, W., Hu, C. H., & Zhou, D. H. (2011).** "Remaining useful life estimation – A review on the statistical data driven approaches." *European Journal of Operational Research*, 213(1), 1-14.
- Comprehensive RUL estimation survey
- **Wiener process** models for degradation
- **Foundation** for our degradation model

**Lei, Y., et al. (2018).** "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." *Mechanical Systems and Signal Processing*, 104, 799-834.
- End-to-end PHM pipeline review
- Feature engineering techniques

### 4.2 Condition-Based Maintenance

**Zhu, J., Chen, N., & Peng, W. (2019).** "Estimation of bearing remaining useful life based on multiscale convolutional neural network." *IEEE Transactions on Industrial Electronics*, 66(4), 3208-3216.
- CNN for RUL estimation from vibration data
- Multi-scale feature extraction

**Jardine, A. K., Lin, D., & Banjevic, D. (2006).** "A review on machinery diagnostics and prognostics implementing condition-based maintenance." *Mechanical Systems and Signal Processing*, 20(7), 1483-1510.
- Classical CBM approaches
- Statistical methods for failure prediction

### 4.3 Joint Scheduling and Maintenance

**Xia, T., et al. (2021).** "Recent advances in prognostics and health management for advanced manufacturing paradigms." *Reliability Engineering & System Safety*, 178, 255-268.
- Integration of PHM with production systems
- **Motivates** health-aware scheduling

**Nguyen, K. A., Do, P., & Grall, A. (2017).** "Joint predictive maintenance and inventory strategy for multi-component systems using Birnbaum's structural importance." *Reliability Engineering & System Safety*, 168, 249-261.
- Joint optimization of maintenance and operations
- Multi-component system modeling

## 5. Proximal Policy Optimization

**Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347*.
- **Core algorithm** used in this work
- Clipped surrogate objective
- Stable policy updates

**Schulman, J., et al. (2016).** "High-dimensional continuous control using generalized advantage estimation." *International Conference on Learning Representations*.
- GAE for variance reduction
- **Used for advantage computation**

**Engstrom, L., et al. (2020).** "Implementation matters in deep RL: A case study on PPO and TRPO." *International Conference on Learning Representations*.
- Implementation details for PPO
- Importance of code-level decisions

## 6. Action Masking in RL

**Huang, S., & Ontañón, S. (2022).** "A closer look at invalid action masking in policy gradient algorithms." *The International FLAIRS Conference Proceedings*, 35.
- Analysis of masking in policy gradients
- Correct probability computation under masks

**Kanervisto, A., & Scheller, C. (2020).** "Action space shaping in deep reinforcement learning." *IEEE Conference on Games*.
- Various approaches to constrained actions
- Comparison of masking vs. penalty methods

## 7. Curriculum Learning

**Bengio, Y., et al. (2009).** "Curriculum learning." *International Conference on Machine Learning*.
- Foundational work on training curricula
- Easier-to-harder task progression

**Narvekar, S., et al. (2020).** "Curriculum learning for reinforcement learning domains: A framework and survey." *Journal of Machine Learning Research*, 21(181), 1-50.
- Comprehensive RL curriculum survey
- Automatic curriculum generation methods

## 8. Multi-Agent and Hierarchical RL

**Foerster, J., et al. (2018).** "Counterfactual multi-agent policy gradients." *AAAI Conference on Artificial Intelligence*.
- Credit assignment in multi-agent systems
- Relevant for multi-equipment coordination

**Kulkarni, T. D., et al. (2016).** "Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation." *Advances in Neural Information Processing Systems*.
- Options framework for hierarchical RL
- **Inspiration** for hierarchical action space

## 9. Industrial Applications of RL

**Hubbs, C. D., et al. (2020).** "A deep reinforcement learning approach for chemical production scheduling." *Computers & Chemical Engineering*, 141, 106982.
- RL for process industry scheduling
- Similar multi-objective formulation

**Leng, J., et al. (2021).** "Digital twin-driven manufacturing cyber-physical system for parallel controlling of smart workshop." *Journal of Ambient Intelligence and Humanized Computing*, 10, 1155-1166.
- Digital twin integration with scheduling
- Real-time optimization framework

## 10. Key Contributions of This Work

Building on the above literature, this work makes the following contributions:

1. **Novel Problem Formulation**: First integrated formulation combining shipyard block scheduling with equipment health management using deep RL.

2. **Heterogeneous GNN Architecture**: Custom 8-edge-type heterogeneous graph representation capturing block-equipment-facility relationships.

3. **Hierarchical Action Masking**: Principled masking approach for complex hierarchical action spaces with precedence constraints.

4. **Health-Aware Rewards**: Multi-objective reward function balancing production efficiency with equipment longevity.

5. **Comprehensive Baseline Comparison**: Systematic evaluation against rule-based, myopic, and siloed optimization approaches.

## References (BibTeX)

```bibtex
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@article{cappart2021combinatorial,
  title={Combinatorial optimization and reasoning with graph neural networks},
  author={Cappart, Quentin and Ch{\'e}telat, Didier and Khalil, Elias and others},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={130},
  pages={1--61},
  year={2021}
}

@article{si2011remaining,
  title={Remaining useful life estimation--A review on the statistical data driven approaches},
  author={Si, Xiao-Sheng and Wang, Wenbin and Hu, Chang-Hua and Zhou, Dong-Hua},
  journal={European Journal of Operational Economics},
  volume={213},
  number={1},
  pages={1--14},
  year={2011}
}

@article{zhang2020learning,
  title={Learning to dispatch for job shop scheduling via deep reinforcement learning},
  author={Zhang, Cong and Song, Wen and Cao, Zhiguang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{velickovic2018graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and others},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```
