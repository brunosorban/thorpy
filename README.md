# Rocket Control Framework

## Description
This repositry comprises two main projects: a trajectory generation framework and a trajectory tracking controller. The trajectory generation framework was developed using differential flatness theory and the trajectory tracking controller was developed using the Model Predictive Control (MPC) framework. The trajectory generation framework is capable of generating trajectories for a rocket with a thrust vectoring system that satisfy the system's dynamics and constraints. The trajectory tracking controller is capable of computing optimal control inputs that track a reference trajectory while satisfying constraints on the system dynamics and control inputs.
For the optimization the CasADi library was adopted. The controller is designed to compute optimal control inputs that track a reference trajectory while satisfying constraints on the system dynamics and control inputs.

The controller is built using the MPC framework, which solves a constrained optimization problem at each time step to compute the optimal control inputs for the next time step. The optimization problem is formulated using CasADi, a powerful tool for numerical optimization and automatic differentiation.

The goal of this project is to derive an optimal controller for a hopper capable to compute safe trajectories, without compromising the system's dynamics and other constraints, such as hardware limits or spatial bounds. The computation of the control commands is based on a performance criterion, currently, minimizing a quadratic cost function that considers position error and energy efficiency. 
<!-- 
## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method. -->

![Animation of the optimal trajectory](Videos/animation.gif)

## Installation
One may install the required packages using the following command:

```bash
pip install hopper_traj
```

Alternatively, one may clone the repository and install the required packages using the following commands:

```bash
pip install -r requirements.txt
```

To install the requirements, and to install the library, one may use the following command:

```bash
pip install -e .
```

## Usage
Examples of both trajectory generation and controller simulation can be found in the examples folder. There are Jupyter Notebooks showing how the workflow was meant to be used.

## Support
For any questions or suggestions, please contact me at:
- Email: bruno.sorban@tum.de or brunosorban@gmail.com

## Roadmap for future contributions
This project is still in development. The next steps are:

- Implement a more realistic model of the system dynamics.
    - Aerodynamics
        - Include the effects of drag and lift.
        - Include the effects of wind.

    - Equations of Motion:
        - Mass variation / Fuel consumption.

- Implement a more realistic simulation environment per se
    - Include the effects of wind.
    - Include the effects of the atmosphere.

## Contributing
Thank you for your interest in contributing to this project! Currently we have no guidelines for external aditions, but feel free to contact me if you have any suggestions or ideas.

## Acknowledgments
I would like to express my gratitude to my mentors, Jon Arrizabalaga and Felix Ebert, for their guidance and support throughout this project. Their expertise and insights have been invaluable in this project and it's results.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Project status
Concluded.
