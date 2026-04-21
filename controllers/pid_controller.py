class PIDController:
    """
    Cascade ACC controller:
    - safety override using TTC / min gap
    - spacing mode when close to target gap
    - speed mode when lead is farther away
    - large-gap recovery mode when the gap becomes too large
    """

    def __init__(
        self,
        dt: float = 1.0,
        d_min: float = 5.0,
        time_headway: float = 0.6,
        max_gap: float = 20.0,
        cruise_speed: float = 19.0,
        k_gap: float = 0.28,
        k_speed: float = 0.35,
        k_rel: float = 0.25,
        ki_speed: float = 0.01,
        accel_max: float = 2.0,
        brake_comfort: float = -2.0,
        brake_strong: float = -3.5,
        emergency_brake: float = -5.0,
        jerk_up_limit: float = 0.8,
        jerk_down_limit: float = 0.8,
        emergency_jerk_limit: float = 2.0,
        ttc_soft: float = 3.5,
        ttc_hard: float = 2.0,
        ref_alpha: float = 0.65,
        gap_deadband: float = 0.25,
        integral_limit: float = 2.0,
        spacing_margin: float = 0.25,
        speed_mode_lead_offset: float = 1.0,
        recovery_gap_buffer: float = 8.0,
        recovery_gain: float = 0.10,
        recovery_speed_offset: float = 2.0,
    ):
        self.dt = dt
        self.d_min = d_min
        self.time_headway = time_headway
        self.max_gap = max_gap
        self.cruise_speed = cruise_speed

        self.k_gap = k_gap
        self.k_speed = k_speed
        self.k_rel = k_rel
        self.ki_speed = ki_speed

        self.accel_max = accel_max
        self.brake_comfort = brake_comfort
        self.brake_strong = brake_strong
        self.emergency_brake = emergency_brake

        self.jerk_up_limit = jerk_up_limit
        self.jerk_down_limit = jerk_down_limit
        self.emergency_jerk_limit = emergency_jerk_limit

        self.ttc_soft = ttc_soft
        self.ttc_hard = ttc_hard

        self.ref_alpha = ref_alpha
        self.gap_deadband = gap_deadband
        self.integral_limit = integral_limit
        self.spacing_margin = spacing_margin
        self.speed_mode_lead_offset = speed_mode_lead_offset

        self.recovery_gap_buffer = recovery_gap_buffer
        self.recovery_gain = recovery_gain
        self.recovery_speed_offset = recovery_speed_offset

        self.prev_accel_cmd = 0.0
        self.prev_v_ref = 0.0
        self.speed_error_int = 0.0

    def reset(self):
        self.prev_accel_cmd = 0.0
        self.prev_v_ref = 0.0
        self.speed_error_int = 0.0

    def compute_target_gap(self, ego_speed: float) -> float:
        target_gap = self.d_min + self.time_headway * ego_speed
        return max(self.d_min, min(self.max_gap, target_gap))

    def _low_pass_ref(self, v_ref_raw: float) -> float:
        v_ref = self.ref_alpha * self.prev_v_ref + (1.0 - self.ref_alpha) * v_ref_raw
        self.prev_v_ref = v_ref
        return v_ref

    def _rate_limit_accel(self, accel_target: float, emergency: bool) -> float:
        if emergency:
            max_delta = self.emergency_jerk_limit * self.dt
        else:
            if accel_target >= self.prev_accel_cmd:
                max_delta = self.jerk_up_limit * self.dt
            else:
                max_delta = self.jerk_down_limit * self.dt

        delta = accel_target - self.prev_accel_cmd

        if delta > max_delta:
            return self.prev_accel_cmd + max_delta
        if delta < -max_delta:
            return self.prev_accel_cmd - max_delta
        return accel_target

    def compute_acceleration(self, ego_speed: float, lead_speed: float, gap: float) -> float:
        relative_speed = lead_speed - ego_speed
        target_gap = self.compute_target_gap(ego_speed)

        closing_speed = ego_speed - lead_speed
        if closing_speed > 1e-6:
            ttc = gap / closing_speed
        else:
            ttc = float("inf")

        emergency = False

        # Primary safety override
        if gap <= self.d_min:
            accel_target = self.emergency_brake
            emergency = True

        elif ttc < self.ttc_hard:
            accel_target = self.emergency_brake
            emergency = True

        elif ttc < self.ttc_soft and gap < max(target_gap, 10.0):
            accel_target = self.brake_strong
            emergency = True

        else:
            gap_error = gap - target_gap
            if abs(gap_error) < self.gap_deadband:
                gap_error = 0.0

            # Large-gap recovery mode
            if gap > target_gap + self.recovery_gap_buffer:
                v_ref_raw = min(
                    self.cruise_speed,
                    lead_speed + self.recovery_speed_offset + self.recovery_gain * gap_error,
                )

            # Normal speed mode
            elif gap > target_gap + self.spacing_margin:
                v_ref_raw = min(self.cruise_speed, lead_speed + self.speed_mode_lead_offset)

            # Spacing mode
            else:
                v_ref_raw = lead_speed + self.k_gap * gap_error
                v_ref_raw = max(0.0, min(self.cruise_speed, v_ref_raw))

            v_ref = self._low_pass_ref(v_ref_raw)

            speed_error = v_ref - ego_speed
            self.speed_error_int += speed_error * self.dt
            self.speed_error_int = max(-self.integral_limit, min(self.integral_limit, self.speed_error_int))

            accel_target = (
                    self.k_speed * speed_error
                    + self.k_rel * relative_speed
                    + self.ki_speed * self.speed_error_int
            )

            accel_target = max(self.brake_comfort, min(self.accel_max, accel_target))

        # Secondary hard safety layer before jerk limiting
        if gap < max(self.d_min + 1.0, 0.75 * target_gap) and closing_speed > 0.5:
            accel_target = min(accel_target, self.brake_strong)

        if gap < self.d_min + 0.5 and closing_speed > 0.2:
            accel_target = min(accel_target, self.emergency_brake)
            emergency = True

        if ttc < self.ttc_hard:
            accel_target = min(accel_target, self.emergency_brake)
            emergency = True
        elif ttc < self.ttc_soft:
            accel_target = min(accel_target, self.brake_strong)

        accel_cmd = self._rate_limit_accel(accel_target, emergency)
        accel_cmd = max(self.emergency_brake, min(self.accel_max, accel_cmd))

        self.prev_accel_cmd = accel_cmd
        return accel_cmd