# Mixed Precision Scientific Machine Learning

The data and code for the paper [J. Hayford, J. Goldman-Wetzler, E. Wang, & L. Lu. Speeding up and reducing memory usage for scientific machine learning via mixed precision. Computer Methods in Applied Mechanics and Engineering,
428,
117093,
2024.](https://www.sciencedirect.com/science/article/pii/S0045782524003499)

## Data
The dataset used for solving the problems associated with this paper.

- DeepONet examples
    - [Advection equation](Dataset/DeepONEt/Advection_equation_dataset)
    - [Linear Instability wave equation](Dataset/DeepONEt/Linear_Instability_Wave_dataset.md)
    Refer to the original paper for this dataset:  [here](https://www.sciencedirect.com/science/article/pii/S0021999122008567)
- PINN examples
    - [Inverse problem of flow in a rectangular domain](Dataset/PINNs/Inverse_problem_of_flow_in_a_rectangular_domain)
    - [Inverse problem of hemodynamics](Dataset/PINNs/Inverse_problem_of_hemodynamics/hemodynamics.md)

## Code

- DeepONet examples
    - [LIW POD DeepONet](DeepOnet/LIW_POD_DeepOnet)
    - [Physics-informed DeepONet](DeepOnet/PI-Diffusion-Reaction-Equation)
    - [Advection equation](DeepOnet/Advection-equation)
- PINN examples
    - [Burgers equation](pinns/dde_burgers_mixed.ipynb)
    - [Kovasznay flow](pinns/Kovasznay_Flow)
    - [Inverse Navier-Stokes](pinns/Navier_Stokes_Inverse)
    - [Hemodynamics](pinns/Hemodynamics)
- Loss landscape and gradient analysis    
    - [Loss landscape](loss-landscape/)
    - [Training phases analysis](training-phases-plots/)
- Theoretical analysis of mixed precision accuracy
    - [Theorem one validation](analysis-of-accuracy/)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{HAYFORD2024117093,
  author  = {Hayford, Joel and Goldman-Wetzler, Jacob and Wang, Eric and Lu, Lu},
  title   = {Speeding up and reducing memory usage for scientific machine learning via mixed precision},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {428},
  pages   = {117093},
  year    = {2024},
  doi     = {https://doi.org/10.1016/j.cma.2024.117093}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
