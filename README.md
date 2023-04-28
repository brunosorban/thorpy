# Rocket Control Framework

## Description
This is a Python implementation of an optimal trajectory controller using the CasADi library. The controller is designed to compute optimal control inputs that track a reference trajectory while satisfying constraints on the system dynamics and control inputs.

The code is organized into several modules:

Function.py: Defines a class to deal with functions in R1.
RK4.py: Implements the Runge-Kutta 4th order numerical integration method for simulating the system dynamics.
Flight.py: Contains a Flight class that defines the simulation environment.
MPC_controller.py: Implements the optimal trajectory controller using the Model Predictive Control (MPC) framework and CasADi optimization.
simulation.py: Sets up the simulation environment using the Flight class and runs the MPC controller to generate the optimal control inputs for the system. (Curently the simulation is being runned inside the controller defined in CasADi)
animate.py: Animates the simulation results using matplotlib and pygame.

The controller is built using the MPC framework, which solves a constrained optimization problem at each time step to compute the optimal control inputs for the next time step. The optimization problem is formulated using CasADi, a powerful tool for numerical optimization and automatic differentiation.

The goal of this project is to derive an optimal controller for a hopper capable to compute safe trajectories, without compromising the system's dynamics and other constraints, such as hardware limits or spatial bounds. The computation of the control commands is based on a performance criterion, currently, minimizing a quadratic cost function that considers position error and energy efficiency. 
<!-- 
## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method. -->

![Animation of the optimal trajectory](https://gitlab.lrz.de/00000000014B2A56/rocket-control-framework/-/blob/main/Videos/animation.gif)


## Installation
Currently, just clone the repository.

## Usage
The simulation can be run via the command line using the following command:

```bash
python simulation.py
```

## Support
For any questions or suggestions, please contact me at:
- Email: bruno.sorban@tum.de or brunosorban@gmail.com

## Roadmap
This project is still in development. The next steps are:

- Implement a more realistic model of the system dynamics.
    - Aerodynamics
        - Include the effects of drag and lift.
        - Include the effects of wind.

    - Equations of Motion:
        - More degrees of freedom.
        - Mass variation / Fuel consumption.

    - Thrust Vectoring system:
        - Include the system as a 1st order dynamics, both from force and moment perspective.

- Implement a more realistic simulation environment per se
    - Include the effects of wind.
    - Include the effects of the atmosphere.

## Contributing
Thank you for your interest in contributing to this project! Currently we have no guidelines for external aditions, but feel free to contact me if you have any suggestions or ideas.

## Acknowledgments
I would like to express my gratitude to my mentors, Jon Arrizabalaga and Felix Ebert, for their guidance and support throughout this project. Their expertise and insights have been invaluable in this project an it's results.

## License
Currenty, no license.

## Project status
Under development.
