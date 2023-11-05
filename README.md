
# Rocket Control Framework

## Description
This repository encompasses two primary projects: a trajectory generation framework and a trajectory tracking controller. The trajectory generation framework is rooted in differential flatness theory, while the trajectory tracking controller leverages the Model Predictive Control (MPC) methodology. Designed to accommodate rockets equipped with thrust vectoring systems, the trajectory generation framework proficiently generates trajectories that adhere to the system's dynamics and constraints. Concurrently, the trajectory tracking controller computes optimal control inputs that faithfully track a reference trajectory while conforming to the system dynamics and control input constraints.

For optimization purposes, the CasADi library has been integrated. This controller, anchored in the MPC framework, systematically solves a constrained optimization problem at each time step to determine the most suitable control inputs for the subsequent time step. Formulated using CasADi, this framework serves as a potent instrument for numerical optimization and automatic differentiation.

The overarching aim of this project is to devise an optimal controller for a hopper, ensuring the computation of safe trajectories without violating the system's dynamics or other constraints, such as hardware limitations or spatial bounds. The control commands are derived based on a performance criterion, presently focused on minimizing a quadratic cost function that accounts for position error and energy efficiency.

## Installation
The necessary packages can be installed using the command:

```bash
pip install hopper_traj
```

Alternatively, one can clone the repository and install the necessary packages using:

```bash
pip install -r requirements.txt
```

To install the requirements and the library, the following command can be utilized:

```bash
pip install -e .
```

## Usage
The examples folder contains instances of both trajectory generation and controller simulation. Jupyter Notebooks are available to demonstrate the intended workflow.

## Support
For inquiries or suggestions, please reach out at:
- Email: bruno.sorban@tum.de or brunosorban@gmail.com

## Roadmap for Future Contributions
This project is a work in progress. Forthcoming developments include:

- Incorporating a more complete model of the system dynamics:
    - Aerodynamics:
        - Integration of drag and lift effects.
        - Inclusion of wind effects.

    - Equations of Motion:
        - Accounting for mass variation/fuel consumption.

- Enhancing the simulation environment:
    - Considering in wind effects.
    - Considering atmospheric influences.

## Contributing
Interest in contributing to this project is greatly appreciated! While there are currently no formal guidelines for external additions, please don't hesitate to reach out if any proposals or ideas are present.

## Acknowledgments
Heartfelt thanks go to the mentors, Jon Arrizabalaga and Felix Ebert, for their invaluable guidance and support throughout this project. Their expertise and insights have been instrumental in achieving the results of this endeavor.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Project Status
Concluded.
