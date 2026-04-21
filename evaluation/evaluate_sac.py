from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

from data_utils import load_speed_profile
from energy_model import get_poly_coeffs


class FullReplayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reference_data: np.ndarray, delta_t: float = 1.0):
        super().__init__()
        self.reference_data = np.asarray(reference_data, dtype=np.float32)
        self.delta_t = float(delta_t)
        self.max_time = len(self.reference_data)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, 0.0, -2.0, 0.0, -np.inf, -4.0], dtype=np.float32),
            high=np.array([50.0, 2.0, 50.0, 2.0, np.inf, np.inf, 4.0], dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.current_time = 0
        self.ego_speed = 0.0
        self.ego_position = 0.0
        self.lead_vehicle_position = 15.0
        self.lead_vehicle_speed = float(self.reference_data[0]) if self.max_time > 0 else 0.0
        self.distance_diff = self.lead_vehicle_position - self.ego_position
        self.previous_accel = 0.0
        self.previous_jerk = 0.0
        self.cumulative_energy_ego = 0.0
        self.cumulative_energy_lead = 0.0
        return self._get_observation(lead_accel=0.0, ego_accel=0.0, jerk=0.0), {}

    def step(self, action: np.ndarray):
        accel = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        jerk = (accel - self.previous_accel) / self.delta_t

        self.ego_speed += accel * self.delta_t
        self.ego_speed = max(self.ego_speed, 0.0)
        self.ego_position += self.ego_speed * self.delta_t

        current_lead_speed = float(self.reference_data[self.current_time])
        if self.current_time < self.max_time - 1:
            next_lead_speed = float(self.reference_data[self.current_time + 1])
            lead_accel = (next_lead_speed - current_lead_speed) / self.delta_t
        else:
            lead_accel = 0.0

        self.lead_vehicle_speed = current_lead_speed
        self.lead_vehicle_position += self.lead_vehicle_speed * self.delta_t
        self.distance_diff = self.lead_vehicle_position - self.ego_position

        power_ego = np.polyval(get_poly_coeffs(self.ego_speed, accel), self.ego_speed)
        power_lead = np.polyval(get_poly_coeffs(self.lead_vehicle_speed, lead_accel), self.lead_vehicle_speed)
        self.cumulative_energy_ego += (power_ego * self.delta_t) / 3600.0
        self.cumulative_energy_lead += (power_lead * self.delta_t) / 3600.0

        done = self.distance_diff < 0.0
        self.current_time += 1
        if self.current_time >= self.max_time:
            done = True

        obs = self._get_observation(lead_accel=lead_accel, ego_accel=accel, jerk=jerk)
        info = {
            "ego_speed": self.ego_speed,
            "lead_speed": self.lead_vehicle_speed,
            "distance_diff": self.distance_diff,
            "accel": accel,
            "lead_accel": lead_accel,
            "jerk": jerk,
            "energy_ego_Wh": self.cumulative_energy_ego,
            "energy_lead_Wh": self.cumulative_energy_lead,
        }
        self.previous_accel = accel
        self.previous_jerk = jerk
        return obs, 0.0, done, False, info

    def _get_observation(self, lead_accel: float = 0.0, ego_accel: float = 0.0, jerk: float = 0.0) -> np.ndarray:
        scaled_energy_diff = (self.cumulative_energy_ego - self.cumulative_energy_lead) / 1000.0
        return np.array(
            [
                self.lead_vehicle_speed,
                lead_accel,
                self.ego_speed,
                ego_accel,
                self.distance_diff,
                scaled_energy_diff * self.max_time,
                jerk,
            ],
            dtype=np.float32,
        )


def save_plot(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / f"{name}.eps", format="eps", bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC ACC model over the full speed profile.")
    parser.add_argument("--velocity_csv", type=Path, default=Path("data/velocity_PID.csv"))
    parser.add_argument("--model_path", type=Path, default=Path("models/sac_custom.zip"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--delta_t", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_speed_profile = load_speed_profile(args.velocity_csv)
    env = FullReplayEnv(reference_data=original_speed_profile, delta_t=args.delta_t)
    model = SAC.load(args.model_path)

    ego_speeds = []
    lead_speeds = []
    distance_diffs = []
    ego_accelerations = []
    lead_accelerations = []
    speed_errors = []
    cumulative_distances_ego = []
    cumulative_distances_lead = []
    cumulative_energy_ego_list = []
    cumulative_energy_lead_list = []
    reference_speeds = []
    ego_vehicle_speeds = []

    cumulative_distance_ego = 0.0
    cumulative_distance_lead = 0.0
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        espeed = info["ego_speed"]
        lspeed = info["lead_speed"]
        dist = info["distance_diff"]
        accel = info["accel"]
        lead_accel = info["lead_accel"]
        e_ego_kwh = info["energy_ego_Wh"] / 1000.0
        e_lead_kwh = info["energy_lead_Wh"] / 1000.0

        cumulative_distance_ego += espeed * args.delta_t
        cumulative_distance_lead += lspeed * args.delta_t

        ego_speeds.append(espeed)
        lead_speeds.append(lspeed)
        distance_diffs.append(dist)
        ego_accelerations.append(accel)
        lead_accelerations.append(lead_accel)
        cumulative_distances_ego.append(cumulative_distance_ego)
        cumulative_distances_lead.append(cumulative_distance_lead)
        cumulative_energy_ego_list.append(e_ego_kwh)
        cumulative_energy_lead_list.append(e_lead_kwh)
        reference_speeds.append(lspeed)
        ego_vehicle_speeds.append(espeed)
        speed_errors.append(lspeed - espeed)

        if truncated:
            done = True

    average_speed = float(np.mean(ego_speeds)) if ego_speeds else 0.0
    mean_speed_error = float(np.mean(speed_errors)) if speed_errors else 0.0
    variance_speed_error = float(np.var(speed_errors)) if speed_errors else 0.0
    mean_acceleration = float(np.mean(ego_accelerations)) if ego_accelerations else 0.0
    variance_acceleration = float(np.var(ego_accelerations)) if ego_accelerations else 0.0
    cumulative_energy_ego = cumulative_energy_ego_list[-1] if cumulative_energy_ego_list else 0.0
    cumulative_energy_lead = cumulative_energy_lead_list[-1] if cumulative_energy_lead_list else 0.0

    print(f"Finished after {step_count} steps.")
    print(f"Average Reward over test period: {total_reward}")
    print(f"Average Speed over test period: {average_speed}")
    print(f"Mean Speed Error: {mean_speed_error} m/s")
    print(f"Variance of Speed Error: {variance_speed_error} (m/s)^2")
    print(f"Mean Acceleration: {mean_acceleration} m/s^2")
    print(f"Variance of Acceleration: {variance_acceleration} (m/s^2)^2")
    print(f"Cumulative Distance (Ego): {cumulative_distance_ego} meters")
    print(f"Cumulative Distance (Lead): {cumulative_distance_lead} meters")
    print(f"Energy (Ego): {cumulative_energy_ego} kWh")
    print(f"Energy (Lead): {cumulative_energy_lead} kWh")
    print(f"Energy Saved: {cumulative_energy_lead - cumulative_energy_ego} kWh")

    speeds_mph = [s * 2.237 for s in ego_vehicle_speeds]
    reference_speeds_mph = [s * 2.237 for s in reference_speeds]
    speed_diff_mph = [ego - ref for ego, ref in zip(speeds_mph, reference_speeds_mph)]
    integral_of_speed_error = np.cumsum(speed_errors) * args.delta_t
    time_axis = np.arange(len(integral_of_speed_error)) * args.delta_t

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_distances_ego, label="Cumulative Distance (Ego)")
    plt.plot(cumulative_distances_lead, label="Cumulative Distance (Lead)")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Distance (meters)")
    plt.title("Cumulative Distance Traveled by Ego and Lead Vehicles")
    plt.legend()
    save_plot(args.output_dir, "cumulative_distance")

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_energy_ego_list, label="Cumulative Energy (Ego) [kWh]")
    plt.plot(cumulative_energy_lead_list, label="Cumulative Energy (Lead) [kWh]")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Energy (kWh)")
    plt.title("Cumulative Energy Consumption by Ego and Lead Vehicles")
    plt.legend()
    save_plot(args.output_dir, "cumulative_energy")

    plt.figure(figsize=(10, 5))
    plt.plot(speeds_mph)
    plt.xlabel("Time Step")
    plt.ylabel("Speed (mph)")
    plt.title("Ego Vehicle Speed over Time")
    save_plot(args.output_dir, "ego_vehicle_speed")

    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds_mph, label="Reference Speed (mph)")
    plt.plot(speeds_mph, label="Ego Vehicle Speed (mph)")
    plt.xlabel("Time Step")
    plt.ylabel("Speed (mph)")
    plt.title("Reference vs Ego Vehicle Speed")
    plt.legend()
    save_plot(args.output_dir, "reference_vs_ego_speed")

    plt.figure(figsize=(10, 5))
    plt.plot(distance_diffs)
    plt.xlabel("Time Step")
    plt.ylabel("Distance Difference (m)")
    plt.title("Distance Difference over Time")
    save_plot(args.output_dir, "distance_difference")

    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds_mph, label="Reference Speed (mph)")
    plt.plot(speeds_mph, label="Ego Vehicle Speed (mph)")
    plt.plot(speed_diff_mph, label="Speed Difference (mph)")
    plt.xlabel("Time Step")
    plt.ylabel("Speed (mph)")
    plt.title("Speed Difference")
    plt.legend()
    save_plot(args.output_dir, "speed_difference")

    plt.figure(figsize=(10, 5))
    plt.plot(speed_errors)
    plt.xlabel("Time Step")
    plt.ylabel("Speed Error (m/s)")
    plt.title("Speed Error over Time")
    save_plot(args.output_dir, "speed_error")

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, integral_of_speed_error, label="Integral of Speed Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral of Speed Error (m*s)")
    plt.title("Integral of Speed Error Over Time")
    plt.legend()
    save_plot(args.output_dir, "integral_speed_error")

    plt.figure(figsize=(10, 5))
    plt.plot(ego_accelerations, label="Ego Vehicle Acceleration (m/s^2)")
    plt.plot(lead_accelerations, label="Lead Vehicle Acceleration (m/s^2)", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Ego and Lead Vehicle Acceleration over Time")
    plt.legend()
    save_plot(args.output_dir, "acceleration_comparison")


if __name__ == "__main__":
    main()
