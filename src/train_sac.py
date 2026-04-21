from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from data_utils import chunk_into_episodes, load_speed_profile, trim_speed_profile
from energy_model import get_poly_coeffs


class VehicleControlEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, reference_data: np.ndarray, max_episode_steps: int = 200, delta_t: float = 1.0):
        super().__init__()
        self.reference_data = np.asarray(reference_data, dtype=np.float32)
        self.delta_t = float(delta_t)
        self.max_episode_steps = int(max_episode_steps)
        self.episodes_list = chunk_into_episodes(
            trim_speed_profile(self.reference_data, self.max_episode_steps), self.max_episode_steps
        )
        self.num_episodes = len(self.episodes_list)
        self.usable_steps = len(self.reference_data)

        self.max_observed_energy_cost = 15000.0
        self.action_space = spaces.Box(low=np.array([-2.0]), high=np.array([2.0]), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, 0.0, -2.0, 0.0, -np.inf, -4.0], dtype=np.float32),
            high=np.array([50.0, 2.0, 50.0, 2.0, np.inf, np.inf, 4.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.episode_index = np.random.randint(0, max(self.num_episodes, 1))
        self.start_index = np.random.randint(0, self.usable_steps - self.max_episode_steps + 1)
        self.chunk_step = 0

        self.lead_vehicle_position = np.random.uniform(5.0, 25.0)
        self.ego_position = 0.0
        self.distance_diff = self.lead_vehicle_position
        self.lead_vehicle_speed = float(self.reference_data[self.start_index])
        self.ego_speed = float(np.random.uniform(0.8 * self.lead_vehicle_speed, 1.2 * self.lead_vehicle_speed))
        self.previous_accel = 0.0
        self.previous_jerk = 0.0
        self.cumulative_energy_cost = 0.0
        self.cumulative_lead_energy_cost = 0.0
        self.accident_occurred = False

        obs = np.array(
            [self.lead_vehicle_speed, 0.0, self.ego_speed, 0.0, self.distance_diff, 0.0, 0.0], dtype=np.float32
        )
        return obs, {}

    def step(self, action: np.ndarray):
        accel = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        jerk = (accel - self.previous_accel) / self.delta_t

        self.ego_speed += accel * self.delta_t
        self.ego_speed = max(self.ego_speed, 0.0)
        self.ego_position += self.ego_speed * self.delta_t + 0.5 * accel * (self.delta_t ** 2)

        lead_idx = self.chunk_step
        current_lead_speed = float(self.reference_data[self.start_index + lead_idx])
        if (self.start_index + lead_idx + 1) < self.usable_steps:
            next_lead_speed = float(self.reference_data[self.start_index + lead_idx + 1])
            lead_accel = (next_lead_speed - current_lead_speed) / self.delta_t
        else:
            lead_accel = 0.0

        self.lead_vehicle_position += current_lead_speed * self.delta_t + 0.5 * lead_accel * (self.delta_t ** 2)
        self.lead_vehicle_speed = current_lead_speed
        self.distance_diff = self.lead_vehicle_position - self.ego_position
        self.accident_occurred = self.distance_diff < 0.0

        poly_coeffs = get_poly_coeffs(self.ego_speed, accel)
        lead_poly_coeffs = get_poly_coeffs(self.lead_vehicle_speed, lead_accel)
        energy_cost = float(np.polyval(poly_coeffs, self.ego_speed))
        lead_energy_cost = float(np.polyval(lead_poly_coeffs, self.lead_vehicle_speed))

        self.cumulative_energy_cost += energy_cost * self.delta_t
        self.cumulative_lead_energy_cost += lead_energy_cost * self.delta_t
        self.max_observed_energy_cost = max(self.max_observed_energy_cost, lead_energy_cost)

        cumulative_energy_difference_wh = (self.cumulative_energy_cost - self.cumulative_lead_energy_cost) / 3600.0
        cumulative_energy_difference_kwh = cumulative_energy_difference_wh / 1000.0

        reward, normalized_distance_reward, normalized_speed_reward, normalized_time_headway_reward, energy_cost_penalty, normalized_smoothness_penalty = self.calculate_reward(
            self.distance_diff,
            self.ego_speed,
            energy_cost,
            lead_energy_cost,
            accel,
            self.previous_accel,
            self.lead_vehicle_speed,
        )

        self.previous_accel = accel
        self.previous_jerk = jerk

        self.episode_index = getattr(self, "episode_index", 0)
        done = False
        episode_progress = self.chunk_step / float(self.max_episode_steps)
        if self.accident_occurred:
            reward += -500.0
            done = True

        self.chunk_step += 1
        if self.chunk_step >= self.max_episode_steps:
            if cumulative_energy_difference_kwh > 0:
                reward -= self.max_episode_steps * abs(cumulative_energy_difference_kwh) * 20.0
            else:
                reward += self.max_episode_steps * abs(cumulative_energy_difference_kwh) * 20.0
            done = True

        observation = np.array(
            [
                self.lead_vehicle_speed,
                lead_accel,
                self.ego_speed,
                accel,
                self.distance_diff,
                self.max_episode_steps * cumulative_energy_difference_kwh * 10.0,
                jerk,
            ],
            dtype=np.float32,
        )
        info = {
            "distance_difference": self.distance_diff,
            "current_speed": self.ego_speed,
            "normalized_distance_reward": normalized_distance_reward,
            "normalized_speed_reward": normalized_speed_reward,
            "normalized_time_headway_reward": normalized_time_headway_reward,
            "energy_cost_penalty": energy_cost_penalty,
            "normalized_smoothness_penalty": normalized_smoothness_penalty,
            "cumulative_energy_difference_Wh": cumulative_energy_difference_wh,
            "episode_progress": episode_progress,
        }
        return observation, reward, done, False, info

    def calculate_reward(
        self,
        distance_diff: float,
        speed: float,
        energy_cost: float,
        lead_energy_cost: float,
        accel: float,
        previous_accel: float,
        ahead_speed: float,
    ) -> Tuple[float, float, float, float, float, float]:
        del lead_energy_cost
        min_distance = 10.0
        max_distance = 30.0
        optimal_distance = (min_distance + max_distance) / 2.0
        energy_cost_weight = 0.0

        def calculate_distance_reward(distance_value: float) -> float:
            distance_mean = (max_distance + min_distance) / 2.0
            desired_interval = 0.5 * (max_distance - min_distance)
            raw_reward = np.exp(-1.0 * ((distance_value - distance_mean) / desired_interval) ** 2)
            return (2.0 * raw_reward) - 1.0

        def adjusted_normal(x: float) -> float:
            mean = 0.0
            std_dev = 1.0 / np.sqrt(np.log(2.0))
            return 5.0 * (np.exp(-5.0 * ((x - mean) / std_dev) ** 2)) - 1.0

        def speed_following_reward(v_desired: float, v_actual: float) -> float:
            return adjusted_normal(v_desired - v_actual)

        def calculate_time_headway(distance_value: float, speed_value: float) -> float:
            if speed_value >= 0.0 and distance_value >= 0.0:
                return distance_value / max(speed_value, 0.005)
            return -abs(distance_value / max(speed_value, 0.005))

        def normalized_gaussian_time_headway_reward(time_headway: float, distance_value: float) -> float:
            time_headway_mean = 2.1
            time_headway_std = 2.0 / np.sqrt(np.log(2.0))
            distance_interval = 0.1 * (max_distance - min_distance)
            distance_mean = optimal_distance
            time_headway_reward = 2.0 * (np.exp(-((time_headway - time_headway_mean) / time_headway_std) ** 2) - 1.0) + 1.0
            raw_distance_reward = np.exp(-0.025 * ((distance_value - distance_mean) / distance_interval) ** 2)
            distance_reward = (2.0 * raw_distance_reward) - 1.0
            if abs(time_headway) > 4.1 or abs(time_headway) < 0.1 or (min_distance < distance_value < max_distance):
                blend_weight = max(0.0, min(1.0, (abs(time_headway) - 0.1) / 4.0))
                return blend_weight * distance_reward + (1.0 - blend_weight) * time_headway_reward
            return time_headway_reward

        def smoothness_reward(current_accel: float, prev_accel: float) -> float:
            jerk_value = (current_accel - prev_accel) / 1.0
            return 1.0 if -0.5 <= jerk_value <= 0.5 else -1.0

        normalized_distance_reward = calculate_distance_reward(distance_diff)
        normalized_speed_reward = speed_following_reward(ahead_speed, speed)
        time_headway = calculate_time_headway(distance_diff, speed)
        normalized_time_headway_reward = normalized_gaussian_time_headway_reward(time_headway, distance_diff)
        normalized_energy_cost = energy_cost / self.max_observed_energy_cost
        energy_cost_penalty = -normalized_energy_cost * energy_cost_weight
        normalized_smoothness_penalty = float(np.clip(smoothness_reward(accel, previous_accel), -1.0, 1.0))

        if speed <= ahead_speed:
            if distance_diff <= 30.0:
                reward = (adjusted_normal(ahead_speed - speed)) * 3.0 / 5.0 + 13.0 / 5.0 + (-0.8) * abs(accel - previous_accel)
            else:
                if accel > 0.0:
                    reward = (1.0 * accel + normalized_distance_reward * 10.0) * 2.0 / 5.0 + (-0.5) * abs(accel - previous_accel)
                else:
                    reward = (1.0 * accel + normalized_distance_reward * 10.0 + self.action_space.low[0]) * 5.0 / 16.0 + (-0.7) * abs(accel - previous_accel)
        else:
            distance_to_collision = (speed**2 - ahead_speed**2) / (2.0 * abs(self.action_space.low[0]))
            if distance_to_collision > distance_diff:
                reward = (2.0 * (-accel + self.action_space.low[0]) - 2.0) * 6.0 / 8.0 - 4.0 / 8.0 + (-0.1) * abs(accel - previous_accel)
            else:
                if distance_diff <= 30.0:
                    reward = (-accel + self.action_space.high[0]) / self.action_space.high[0] + 1.0 + (-1.0) * abs(accel - previous_accel)
                else:
                    if accel > 0.0:
                        reward = (1.0 * accel + normalized_distance_reward * 10.0) / 2.0 - 1.0 / 2.0 + (-0.7) * abs(accel - previous_accel)
                    else:
                        reward = (1.0 * accel + normalized_distance_reward * 10.0 + self.action_space.low[0]) * 5.0 / 16.0 + (-0.8) * abs(accel - previous_accel)

        return (
            float(reward),
            float(normalized_distance_reward),
            float(normalized_speed_reward),
            float(normalized_time_headway_reward),
            float(energy_cost_penalty),
            float(normalized_smoothness_penalty),
        )


class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir: Path, log_name: str = "training_log.csv", verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_path = self.log_dir / log_name
        self.episode_rewards = []
        self.episode_metrics = {
            "normalized_distance_reward": [],
            "normalized_speed_reward": [],
            "normalized_time_headway_reward": [],
            "energy_cost_penalty": [],
            "smoothness_penalty": [],
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "timestep",
                "average_reward",
                "normalized_distance_reward",
                "normalized_speed_reward",
                "normalized_time_headway_reward",
                "energy_cost_penalty",
                "normalized_smoothness_penalty",
            ])

    def _on_step(self) -> bool:
        timestep = self.num_timesteps
        reward = self.locals.get("rewards", [0])[-1]
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        self.episode_rewards.append(reward)
        self.episode_metrics["normalized_distance_reward"].append(info.get("normalized_distance_reward", 0))
        self.episode_metrics["normalized_speed_reward"].append(info.get("normalized_speed_reward", 0))
        self.episode_metrics["normalized_time_headway_reward"].append(info.get("normalized_time_headway_reward", 0))
        self.episode_metrics["energy_cost_penalty"].append(info.get("energy_cost_penalty", 0))
        self.episode_metrics["smoothness_penalty"].append(info.get("normalized_smoothness_penalty", 0))

        if self.locals.get("dones", [False])[-1]:
            averages = {k: float(np.mean(v)) if v else 0.0 for k, v in self.episode_metrics.items()}
            average_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            with self.log_path.open("a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    timestep,
                    average_reward,
                    averages["normalized_distance_reward"],
                    averages["normalized_speed_reward"],
                    averages["normalized_time_headway_reward"],
                    averages["energy_cost_penalty"],
                    averages["smoothness_penalty"],
                ])
            self.logger.record("reward/average_reward", average_reward)
            self.logger.record("reward/normalized_distance_reward", averages["normalized_distance_reward"])
            self.logger.record("reward/normalized_speed_reward", averages["normalized_speed_reward"])
            self.logger.record("reward/normalized_time_headway_reward", averages["normalized_time_headway_reward"])
            self.logger.record("reward/energy_cost_penalty", averages["energy_cost_penalty"])
            self.logger.record("reward/smoothness_penalty", averages["smoothness_penalty"])
            self.episode_rewards.clear()
            for values in self.episode_metrics.values():
                values.clear()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC for ACC with EV energy-aware dynamics.")
    parser.add_argument("--velocity_csv", type=Path, default=Path("data/velocity_PID.csv"))
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--total_timesteps", type=int, default=1880 * 1500)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--delta_t", type=float, default=1.0)
    parser.add_argument("--clear_log_dir", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Running on device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print(f"CUDA version: {torch.version.cuda}")

    speed_profile = load_speed_profile(args.velocity_csv)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_log_dir:
        for filename in args.log_dir.iterdir():
            if filename.is_file() or filename.is_symlink():
                filename.unlink()

    logger = configure(str(args.log_dir), ["stdout", "tensorboard"])

    env = make_vec_env(
        lambda: gymnasium.wrappers.TimeLimit(
            VehicleControlEnv(speed_profile, max_episode_steps=args.max_episode_steps, delta_t=args.delta_t),
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=args.n_envs,
    )

    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU, normalize_images=False)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        batch_size=256,
        buffer_size=2_000_000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        device=str(device),
    )
    model.set_logger(logger)

    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.max_episode_steps,
        callback=CustomLoggingCallback(args.log_dir),
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    model.save(args.log_dir / "sac_custom")
    print(f"Saved trained model to: {args.log_dir / 'sac_custom.zip'}")


if __name__ == "__main__":
    main()
