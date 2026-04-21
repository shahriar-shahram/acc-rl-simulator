import matplotlib.pyplot as plt
import numpy as np
from controllers.pid_controller import PIDController
from env.acc_env import ACCEnvironment
from pathlib import Path
def run_simulation(
    scenario_path: str = "scenarios/real/FTP75.csv",
    dt: float = 1.0,
):
    scenario_name = Path(scenario_path).stem
    env = ACCEnvironment(
        scenario_path=scenario_path,
        dt=dt,
        initial_gap=20.0,
        initial_ego_speed=None,
    )
    print("Scenario path:", scenario_path)
    print("Loaded samples:", len(env.lead_profile))
    print("dt:", dt)
    print("Expected duration [s]:", len(env.lead_profile) * dt)

    controller = PIDController(
        dt=dt,
        d_min=10.0,
        # time_headway=0.6,
        time_headway=1.0,

        max_gap=30.0,
        # cruise_speed=19.0,
        cruise_speed=30.0,
        # k_gap=0.32,
        k_gap=0.6,
        # k_speed=0.28,
        k_speed=0.02,

        # k_rel=0.32,
        k_rel=1.0,

        ki_speed=0.01,
        # accel_max=2.0,
        accel_max=2.0,
        # brake_comfort=-2.0,
        # brake_strong=-3.5,
        # emergency_brake=-5.0,

        brake_comfort=-1.5,
        brake_strong=-2.0,
        emergency_brake=-3.0,

        jerk_up_limit=0.8,
        jerk_down_limit=0.8,
        emergency_jerk_limit=1.5,
        # ttc_soft=4.5,
        # ttc_hard=2.8,
        ttc_soft=2.0,
        ttc_hard=1.0,
        # ref_alpha=0.65,
        ref_alpha=0.5,
        gap_deadband=0.25,
        integral_limit=2.0,
        spacing_margin=0.6,
        speed_mode_lead_offset=7.0,
        recovery_gap_buffer=8.0,
        recovery_gain=0.10,
        recovery_speed_offset=2.0,
    )

    state = env.reset()
    controller.reset()

    times = []
    lead_speeds = []
    ego_speeds = []
    gaps = []
    desired_gaps = []
    accelerations = []
    jerks = []

    collision_flag = False

    while True:
        # current state
        t = state["time"]
        lead_speed = state["lead_speed"]
        ego_speed = state["ego_speed"]
        gap = state["gap"]

        # record current state
        times.append(t)
        lead_speeds.append(lead_speed)
        ego_speeds.append(ego_speed)
        gaps.append(gap)
        desired_gaps.append(controller.compute_target_gap(ego_speed))
        accelerations.append(state["acceleration"])
        jerks.append(state["jerk"])

        if state["collision"]:
            collision_flag = True

        # control action
        accel_cmd = controller.compute_acceleration(
            ego_speed=ego_speed,
            lead_speed=lead_speed,
            gap=gap,
        )

        # step environment
        next_state, done = env.step(accel_cmd)

        # if terminal, record the terminal state too
        if done:
            times.append(next_state["time"])
            lead_speeds.append(next_state["lead_speed"])
            ego_speeds.append(next_state["ego_speed"])
            gaps.append(next_state["gap"])
            desired_gaps.append(controller.compute_target_gap(next_state["ego_speed"]))
            accelerations.append(next_state["acceleration"])
            jerks.append(next_state["jerk"])

            if next_state["collision"]:
                collision_flag = True

            break

        state = next_state

    # Convert to NumPy for easier metrics calculation
    times = np.array(times)
    gaps = np.array(gaps)
    ego_speeds = np.array(ego_speeds)
    lead_speeds = np.array(lead_speeds)
    accelerations = np.array(accelerations)
    jerks = np.array(jerks)

    # Calculate final metrics
    min_gap = np.min(gaps)
    mean_abs_speed_error = np.mean(np.abs(lead_speeds - ego_speeds))
    mean_abs_jerk = np.mean(np.abs(jerks))
    max_abs_jerk = np.max(np.abs(jerks))

    total_steps = len(gaps)
    pct_below_dmin = np.mean(gaps < controller.d_min) * 100
    pct_between_band = np.mean((gaps >= controller.d_min) & (gaps <= controller.max_gap)) * 100
    pct_above_maxgap = np.mean(gaps > controller.max_gap) * 100

    print("Stored time samples:", len(times))
    print("Last time value:", times[-1] if len(times) else None)

    print("\nSimulation Metrics")
    print("------------------")
    print(f"min_gap: {min_gap}")
    print(f"collision: {collision_flag}")
    print(f"pct_below_{int(controller.d_min)}m: {pct_below_dmin}")
    print(f"pct_between_{int(controller.d_min)}m_and_{int(controller.max_gap)}m: {pct_between_band}")
    print(f"pct_above_{int(controller.max_gap)}m: {pct_above_maxgap}")
    print(f"mean_abs_speed_error: {mean_abs_speed_error}")
    print(f"mean_abs_jerk: {mean_abs_jerk}")
    print(f"max_abs_jerk: {max_abs_jerk}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(times, lead_speeds, label="Lead Speed")
    plt.plot(times, ego_speeds, label="Ego Speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Lead vs Ego Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"assets/{scenario_name}_speed_tracking.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, gaps, label="Actual Gap")
    plt.plot(times, desired_gaps, label="Desired Gap")
    plt.axhline(controller.d_min, color="red", linestyle="--", label=f"Min Safe Gap ({controller.d_min:.0f} m)")
    plt.axhline(controller.max_gap, color="green", linestyle="--",label=f"Upper Gap Target ({controller.max_gap:.0f} m)")
    plt.xlabel("Time [s]")
    plt.ylabel("Gap [m]")
    plt.title("Actual vs Desired Gap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"assets/{scenario_name}_gap_tracking.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, accelerations)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Ego Acceleration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"assets/{scenario_name}_acceleration.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(times, jerks)
    plt.xlabel("Time [s]")
    plt.ylabel("Jerk [m/s³]")
    plt.title("Ego Jerk")
    plt.axhline(0.8, linestyle="--", color='red', label="Comfort Limit (0.8)")
    plt.axhline(-0.8, linestyle="--", color='red')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"assets/{scenario_name}_jerk.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import sys

    scenario = "scenarios/real/FTP75.csv"
    if len(sys.argv) > 1:
        scenario = sys.argv[1]

    run_simulation(scenario)