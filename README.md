# ACC RL Simulator

Adaptive Cruise Control (ACC) simulator for comparing a classical jerk-limited PID baseline against future learning-based controllers on real lead-vehicle speed profiles.

This project currently includes:
- a discrete-time longitudinal vehicle simulation
- a real drive-cycle loader using uploaded speed traces
- a jerk-limited PID-based ACC baseline
- evaluation metrics and plots for speed tracking, spacing, acceleration, and jerk

The current baseline has been tuned and tested on the FTP75 scenario and is intended to serve as the classical benchmark before adding reinforcement learning.

---

## Project Goal

The goal of this project is to build an interactive and cloud-deployable ACC simulator where:
- the **lead vehicle** follows a real speed profile
- the **ego vehicle** is controlled by a baseline controller or, later, an RL policy
- performance is evaluated in terms of:
  - collision avoidance
  - spacing behavior
  - speed tracking
  - acceleration smoothness
  - jerk

The broader plan is to compare:
1. a classical baseline controller
2. an RL controller trained under the same dynamics and scenarios

---

## Current Status

### Implemented
- Real drive-cycle scenario loading from CSV
- Constant-acceleration longitudinal ego vehicle model
- Lead vehicle position propagation from measured speed trace
- Jerk-limited PID-style cascade ACC baseline
- Full-run simulation and metric reporting
- Plot generation for:
  - lead vs ego speed
  - actual vs desired gap
  - ego acceleration
  - ego jerk

### Not implemented yet
- RL training and inference
- Streamlit interface
- cloud deployment
- multi-scenario batch evaluation table
- MPC baseline

---

## Repository Structure

```text
acc-rl-simulator/
├── app/
│   └── streamlit_app.py
├── controllers/
│   ├── pid_controller.py
│   └── rl_controller.py
├── env/
│   └── acc_env.py
├── evaluation/
│   └── simulate.py
├── training/
│   └── train_sac.py
├── scenarios/
│   └── real/
│       ├── FTP75.csv
│       ├── MetroHighwayCal.csv
│       └── NRELClassElectricVehicleCycle.csv
├── assets/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Vehicle and Simulation Model

The simulator uses a simple longitudinal kinematic model.

### Lead vehicle
The lead vehicle speed is read from a CSV file and converted from **mph to m/s**. Its position is updated as:

\[
x_{\text{lead},k+1} = x_{\text{lead},k} + v_{\text{lead},k}\Delta t
\]

### Ego vehicle
The ego vehicle uses a constant-acceleration update:

\[
x_{\text{ego},k+1} = x_{\text{ego},k} + v_{\text{ego},k}\Delta t + \frac{1}{2}a_k \Delta t^2
\]

\[
v_{\text{ego},k+1} = \max(0,\; v_{\text{ego},k} + a_k \Delta t)
\]

where:
- \(x\) is longitudinal position
- \(v\) is longitudinal speed
- \(a\) is commanded acceleration
- \(\Delta t = 1.0\) s in the current setup

The spacing gap is:

\[
gap_k = x_{\text{lead},k} - x_{\text{ego},k}
\]

Jerk is computed as:

\[
j_k = \frac{a_k - a_{k-1}}{\Delta t}
\]

---

## Current Baseline Controller

The current classical baseline is a **jerk-limited PID-style cascade ACC controller** with:
- spacing-based target gap
- speed-mode / spacing-mode switching
- large-gap recovery behavior
- TTC-based safety override
- jerk-limited acceleration commands

### Target Gap Policy

The controller uses a dynamic desired gap:

\[
d_{\text{target}} = \mathrm{clip}(d_{\min} + h\,v_{\text{ego}},\; d_{\min},\; d_{\max})
\]

The current tuned baseline uses:
- minimum gap: **10 m**
- maximum gap: **30 m**
- time headway: **1.0 s**

### Current Baseline Parameters

```python
controller = PIDController(
    dt=dt,
    d_min=10.0,
    time_headway=1.0,
    max_gap=30.0,
    cruise_speed=30.0,
    k_gap=0.6,
    k_speed=0.02,
    k_rel=1.0,
    ki_speed=0.01,
    accel_max=2.0,
    brake_comfort=-1.5,
    brake_strong=-2.0,
    emergency_brake=-3.0,
    jerk_up_limit=0.8,
    jerk_down_limit=0.8,
    emergency_jerk_limit=1.5,
    ttc_soft=2.0,
    ttc_hard=1.0,
    ref_alpha=0.5,
    gap_deadband=0.25,
    integral_limit=2.0,
    spacing_margin=0.6,
    speed_mode_lead_offset=7.0,
    recovery_gap_buffer=8.0,
    recovery_gain=0.10,
    recovery_speed_offset=2.0,
)
```

### Controller Interpretation

- `d_min`, `max_gap`, `time_headway`: define the desired operating spacing band
- `k_gap`: spacing error influence on reference generation
- `k_speed`: speed-reference tracking term
- `k_rel`: relative speed damping / car-following responsiveness
- `accel_max`, `brake_*`: longitudinal comfort and safety limits
- `jerk_*`: smoothness limits
- `ttc_soft`, `ttc_hard`: time-to-collision based safety thresholds
- `ref_alpha`: low-pass filtering on the reference speed

---

## Scenario Data

The current simulator uses real lead-vehicle speed traces stored in:

- `scenarios/real/FTP75.csv`
- `scenarios/real/MetroHighwayCal.csv`
- `scenarios/real/NRELClassElectricVehicleCycle.csv`

The current evaluation focus has been on:
- **FTP75**

---

## Current FTP75 Baseline Result

Representative result for the current baseline on FTP75:

- **collision:** False
- **min gap:** 7.1074 m
- **pct below 10 m:** 2.13%
- **pct between 10 m and 30 m:** 73.10%
- **pct above 30 m:** 24.77%
- **mean abs speed error:** 0.4107 m/s
- **mean abs jerk:** 0.2074 m/s³
- **max abs jerk:** 1.5 m/s³

This controller does not strictly satisfy a hard 10 m minimum at every instant, but it provides a stable, interpretable, and collision-free baseline with good tracking and moderate comfort.

---

## How to Run

Run the current evaluation with:

```bash
python -m evaluation.simulate
```

The script:
- loads the selected real scenario
- runs the PID baseline over the full trace
- prints summary metrics
- saves plots to the `assets/` folder

---

## Generated Outputs

The current evaluation generates:
- speed tracking plot
- gap tracking plot
- acceleration plot
- jerk plot

Typical saved files include:
- `assets/ftp75_speed_tracking_latest.png`
- `assets/ftp75_gap_tracking_latest.png`
- `assets/ftp75_acceleration_latest.png`
- `assets/ftp75_jerk_latest.png`

---

## Planned Next Steps

1. Freeze the current PID baseline as the classical benchmark
2. Implement the RL environment and reward design
3. Train an RL controller under the same dynamics and scenario setup
4. Compare RL against the PID baseline across:
   - safety
   - spacing compliance
   - speed tracking
   - acceleration smoothness
   - jerk
5. Add Streamlit visualization and later deploy the app to the cloud

---

## Notes

- The current controller is tuned for practical baseline performance rather than formal optimality.
- The present focus is on building a reliable comparison framework before adding RL.
- Future versions may add:
  - RL controllers
  - multi-scenario evaluation tables
  - Streamlit dashboard
  - cloud deployment
  - MPC baseline
