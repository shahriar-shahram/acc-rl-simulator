import csv
import math
from pathlib import Path


def generate_highway_profile(
    output_path: str = "scenarios/generated/highway.csv",
    dt: float = 0.1,
    total_time: float = 60.0,
) -> None:
    """
    Generate a smooth highway lead-vehicle speed profile.

    Speed is in m/s.
    Time is in seconds.
    """

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "lead_speed"])

        t = 0.0
        while t <= total_time + 1e-9:
            # Base cruise speed: about 22 m/s (~49 mph)
            v = 22.0

            # Smooth background traffic-flow variations
            v += 1.2 * math.sin(0.12 * t)
            v += 0.6 * math.sin(0.035 * t + 0.8)

            # One moderate slowdown event around 30 s
            slowdown = 4.5 * math.exp(-((t - 30.0) / 5.5) ** 2)
            v -= slowdown

            # Keep the speed physically reasonable
            v = max(12.0, v)

            writer.writerow([round(t, 2), round(v, 3)])
            t += dt

    print(f"Created {out_path}")


if __name__ == "__main__":
    generate_highway_profile()
