# Mixed Precision Scientific Machine Learning

The source code and data for the article [Speeding up and reducing memory usage for scientific machine learning via mixed precision](https://arxiv.org/abs/2401.16645).

## Data
The dataset used for solving the problems associated with this paper. Some of the datasets are available on this repository. Others are available on google drive through the attached link.

### DeepONet
- [Advection equation](Dataset/DeepONEt/Advection_equation_dataset)
- [Linear Instability wave equation](Dataset/DeepONEt/Linear_Instability_Wave_dataset.md)
Refer to the original paper for this dataset:  
[URL: https://www.sciencedirect.com/science/article/pii/S0021999122008567](https://www.sciencedirect.com/science/article/pii/S0021999122008567)

### PINNs
- [Inverse problem of flow in a rectangular domain](Dataset/PINNs/Inverse_problem_of_flow_in_a_rectangular_domain)
- [Inverse problem of hemodynamics](Dataset/PINNs/Inverse_problem_of_hemodynamics/hemodynamics.md)
## Code

- [Loss landscape and gradient analysis](loss-landscape/)
- [Burgers equation](pinns/dde_burgers_mixed.ipynb)
- [Inverse Navier-Stokes](pinns/Navier_Stokes_Inverse)
- [Kovasznay flow](pinns/Kovasznay_Flow)
- [Hemodynamics](pinns/Hemodynamics)
- [POD DeepONet](DeepOnet/LIW_POD_DeepOnet)
- [Physics-informed DeepONet](DeepOnet/PI-Diffusion-Reaction-Equation)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{...,
  title   = {Speeding up and reducing memory usage for scientific machine learning via mixed precision},
  author  = {Hayford, Joel and Goldman-Wetzler, Jacob and Wang, Eric and Lu, Lu},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2024},
  ...
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
