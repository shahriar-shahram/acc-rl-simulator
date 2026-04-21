import pandas as pd


class ACCEnvironment:
    def __init__(
        self,
        scenario_path: str,
        dt: float = 1.0,
        initial_gap: float = 15.0,
        initial_ego_speed=None,
    ):
        self.scenario_path = scenario_path
        self.dt = dt
        self.initial_gap = initial_gap
        self.initial_ego_speed = initial_ego_speed

        self.lead_profile = self._load_scenario()
        self.reset()

    def _load_scenario(self):
        df = pd.read_csv(self.scenario_path)

        if df.shape[1] == 1:
            lead_speed = df.iloc[:, 0].astype(float).to_list()
        else:
            lead_speed = df.iloc[:, 1].astype(float).to_list()

        # Dataset is in mph -> convert to m/s
        lead_speed = [v * 0.44704 for v in lead_speed]
        return lead_speed

    def reset(self):
        self.step_idx = 0

        self.time = 0.0
        self.lead_pos = self.initial_gap
        self.ego_pos = 0.0

        self.lead_speed = self.lead_profile[0]
        self.ego_speed = self.lead_profile[0] if self.initial_ego_speed is None else self.initial_ego_speed

        self.ego_accel = 0.0
        self.prev_ego_accel = 0.0

        self.gap = self.lead_pos - self.ego_pos
        self.jerk = 0.0
        self.collision = False

        return self._get_state()

    def _get_state(self):
        relative_speed = self.lead_speed - self.ego_speed
        return {
            "time": self.time,
            "lead_speed": self.lead_speed,
            "ego_speed": self.ego_speed,
            "relative_speed": relative_speed,
            "gap": self.gap,
            "acceleration": self.ego_accel,
            "jerk": self.jerk,
            "collision": self.collision,
        }

    def step(self, ego_accel: float):
        if self.step_idx >= len(self.lead_profile) - 1:
            return self._get_state(), True

        self.prev_ego_accel = self.ego_accel
        self.ego_accel = ego_accel

        # Lead vehicle update
        self.lead_speed = self.lead_profile[self.step_idx]
        self.lead_pos += self.lead_speed * self.dt

        # Ego vehicle update using constant-acceleration kinematics
        ego_speed_old = self.ego_speed
        self.ego_pos += ego_speed_old * self.dt + 0.5 * self.ego_accel * (self.dt ** 2)
        self.ego_speed = max(0.0, ego_speed_old + self.ego_accel * self.dt)

        # Derived quantities
        self.gap = self.lead_pos - self.ego_pos
        self.jerk = (self.ego_accel - self.prev_ego_accel) / self.dt

        if self.gap <= 0.0:
            self.collision = True

        self.step_idx += 1
        self.time = self.step_idx * self.dt

        done = self.collision or (self.step_idx >= len(self.lead_profile) - 1)
        return self._get_state(), done
