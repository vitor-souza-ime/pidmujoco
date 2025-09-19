# PID Adaptive Control for 3-DOF Robot Simulation in MuJoCo

This repository contains a Python simulation of a simple 3-DOF robotic manipulator controlled using an adaptive PID controller. The simulation is implemented using [MuJoCo](https://mujoco.org/) and visualized with Matplotlib. The goal is to demonstrate the performance of an adaptive PID controller with anti-windup, gravity compensation, and gain adaptation.

## Repository

[https://github.com/vitor-souza-ime/pidmujoco](https://github.com/vitor-souza-ime/pidmujoco)

## Features

- 3-DOF robotic manipulator modeled in MuJoCo.
- Adaptive PID controller with:
  - Proportional-Integral-Derivative (PID) control
  - Anti-windup for integral term
  - Gravity compensation
  - Conservative automatic gain adaptation
- Smooth reference trajectory generation for joints.
- Data logging and visualization of:
  - Joint positions vs desired trajectories
  - RMS tracking error
  - PID gain adaptation over time

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vitor-souza-ime/pidmujoco.git
cd pidmujoco
````

2. Install dependencies (recommended via `pip`):

```bash
pip install mujoco numpy matplotlib
```

> Ensure that MuJoCo is installed and configured correctly. You may need a license key or use the free trial version.

## Usage

Run the simulation using:

```bash
python main.py
```

The script will:

1. Create a simple 3-DOF robot model.
2. Execute a 20-second simulation with adaptive PID control.
3. Generate and save plots (`adaptive_control_results.png`) showing:

   * Trajectory tracking
   * RMS error
   * Gain adaptation
4. Print key performance metrics to the console.

## Output

The main output includes:

* `adaptive_control_results.png`: A 2x2 figure summarizing the performance.
* Console logs reporting:

  * Initial and final RMS tracking error
  * Improvement percentage
  * Simulation time and number of data points

## File Structure

```
pidmujoco/
│
├─ main.py               # Simulation and adaptive PID control
├─ README.md             # This documentation
└─ adaptive_control_results.png  # Generated figure after running the script
```

## References

* [MuJoCo Physics Engine](https://mujoco.org/)
* PID Control and Adaptive Control Literature

## License

MIT License © Vitor Amadeu Souza

