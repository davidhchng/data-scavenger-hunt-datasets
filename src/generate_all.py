import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED = 42


def _format_timestamp(dt: datetime, fmt_choice: int) -> str:
    if fmt_choice == 0:
        # YYYY-MM-DD
        return dt.strftime("%Y-%m-%d")
    if fmt_choice == 1:
        # YYYY/MM/DD
        return dt.strftime("%Y/%m/%d")
    if fmt_choice == 2:
        # YYYY-MM-DD HH:MM (24h)
        return dt.strftime("%Y-%m-%d %H:%M")
    if fmt_choice == 3:
        # YYYY-MM-DD HH:MMAM (12h, no space) e.g. 2025-03-04 11:10AM
        return dt.strftime("%Y-%m-%d %I:%M%p")
    if fmt_choice == 4:
        # YYYY-MM-DD HH:MM:SS (24h)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if fmt_choice == 5:
        # YYYY-MM-DD HH:MM:SS.xx (fractional seconds)
        # We'll generate .00 to .99 explicitly for realism.
        frac = np.random.randint(0, 100)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{frac:02d}"

    # Fallback
    return dt.strftime("%Y-%m-%d %H:%M")


def generate_weather_dataset(out_path: str) -> None:
    np.random.seed(SEED)

    n = 3000
    station_id = "YK-01"

    start = datetime(2025, 1, 1, 0, 0)
    step = timedelta(hours=4)

    base_dts = [start + i * step for i in range(n)]

    minutes = np.random.randint(0, 60, size=n)

    has_seconds = np.random.rand(n) < 0.25
    seconds = np.where(has_seconds, np.random.randint(0, 60, size=n), 0)

    dts = [
        dt.replace(minute=int(m), second=int(s), microsecond=0)
        for dt, m, s in zip(base_dts, minutes, seconds)
]

    temps = []
    for dt in dts:
        day_of_year = dt.timetuple().tm_yday
        seasonal = 18.0 * np.sin(2 * np.pi * (day_of_year / 365.25))
        noise = np.random.normal(0, 6.0)
        temp = -12.0 + seasonal + noise
        temps.append(temp)
    temp_c = np.clip(np.array(temps), -45, 25).round(1)

    humidity_percent = np.random.randint(50, 96, size=n)  # 50..95
    wind_kmh = np.random.gamma(shape=2.0, scale=4.0, size=n)  # right-skew
    wind_kmh = np.clip(wind_kmh, 0, 40).round(0).astype(int)

    fmt_choices = np.random.randint(0, 6, size=n)  # 0..5 formats
    is_date_only = np.random.rand(n) < 0.30

    timestamps = []
    for dt, fmt, date_only in zip(dts, fmt_choices, is_date_only):
        if date_only:
            timestamps.append(dt.strftime("%Y-%m-%d") if (np.random.rand() < 0.6) else dt.strftime("%Y/%m/%d"))
        else:
            timestamps.append(_format_timestamp(dt, int(fmt)))

    df = pd.DataFrame({
        "station_id": [station_id] * n,
        "timestamp": timestamps,
        "temp_c": temp_c,
        "humidity_percent": humidity_percent,
        "wind_kmh": wind_kmh,
    })

    df.to_csv(out_path, index=False)


def generate_coffee_dataset(out_path: str) -> None:

    np.random.seed(SEED)

    n_each = 50
    faculties = (["Engineering"] * n_each) + (["Science"] * n_each)
    student_id = np.arange(1, 2 * n_each + 1)

    year = np.random.randint(1, 5, size=2 * n_each)  # 1..4

    eng = np.random.normal(loc=3.0, scale=1.0, size=n_each)
    sci = np.random.normal(loc=2.5, scale=1.0, size=n_each)
    coffee = np.concatenate([eng, sci])

    coffee = np.clip(coffee, 0, 8)
    coffee = np.round(coffee * 2) / 2

    df = pd.DataFrame({
        "student_id": student_id,
        "faculty": faculties,
        "year": year,
        "coffee_cups_per_day": coffee,
    })

    df.to_csv(out_path, index=False)


def generate_grades_dataset(out_path: str) -> None:
    np.random.seed(SEED)

    n_per = 200
    courses = (["MATH100"] * n_per) + (["STAT200"] * n_per) + (["COMM293"] * n_per)
    student_id = np.arange(1, 3 * n_per + 1)

    math = np.random.normal(loc=64, scale=15, size=n_per)
    stat = np.random.normal(loc=72, scale=10, size=n_per)
    comm = np.random.beta(a=8, b=2, size=n_per) * 100  # skew toward high

    grades = np.concatenate([math, stat, comm])
    grades = np.clip(grades, 0, 100).round(0).astype(int)

    df = pd.DataFrame({
        "student_id": student_id,
        "course": courses,
        "grade": grades,
    })

    df.to_csv(out_path, index=False)


def save_grades_histogram(grades_csv_path: str, out_path: str) -> None:
    df = pd.read_csv(grades_csv_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bins = np.arange(0, 101, 5)

    plt.figure(figsize=(10, 6))
    for course in ["MATH100", "STAT200", "COMM293"]:
        subset = df.loc[df["course"] == course, "grade"]
        plt.hist(subset, bins=bins, alpha=0.5, label=course)

    plt.title("Grade Distributions by Course")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def save_weather_preview(weather_csv_path: str, out_path: str) -> None:
    """
    Useful preview for weather dataset:
    - temp over time (after parsing timestamps)
    - makes the dataset feel real and sanity-checks parsing
    """
    df = pd.read_csv(weather_csv_path)

    # Robust parse: infer formats; date-only will become midnight here (that's fine for preview)
    # Participants will do their own "assume noon" logic in the challenge.
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
    # second pass for any stragglers (often helps with mixed formats)
    dt2 = pd.to_datetime(df.loc[dt.isna(), "timestamp"], errors="coerce", format="mixed")
    dt.loc[dt.isna()] = dt2

    preview = df.copy()
    preview["parsed_timestamp"] = dt
    preview = preview.dropna(subset=["parsed_timestamp"]).sort_values("parsed_timestamp")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(preview["parsed_timestamp"], preview["temp_c"])
    plt.title("Weather Station Preview: Temperature Over Time (parsed)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_coffee_preview(coffee_csv_path: str, out_path: str) -> None:
    """
    Useful preview for coffee dataset:
    - side-by-side boxplots of coffee by faculty
    - immediately supports the hypothesis test story
    """
    df = pd.read_csv(coffee_csv_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    eng = df.loc[df["faculty"] == "Engineering", "coffee_cups_per_day"]
    sci = df.loc[df["faculty"] == "Science", "coffee_cups_per_day"]

    plt.figure(figsize=(8, 6))
    plt.boxplot([eng, sci], labels=["Engineering", "Science"])
    plt.title("Coffee Consumption by Faculty")
    plt.ylabel("Coffee (cups/day)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grades_overlay_histogram(grades_csv_path: str, out_path: str) -> None:
    """
    Overlay histogram (what you already have conceptually, but as a clearly named function).
    """
    df = pd.read_csv(grades_csv_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bins = np.arange(0, 101, 5)

    plt.figure(figsize=(10, 6))
    for course in ["MATH100", "STAT200", "COMM293"]:
        subset = df.loc[df["course"] == course, "grade"]
        plt.hist(subset, bins=bins, alpha=0.5, label=course)

    plt.title("Grade Distributions by Course (Overlay Histogram)")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grades_faceted_histograms(grades_csv_path: str, out_path: str) -> None:
    """
    Separate (faceted) histograms: 3 subplots, one per course.
    Cleaner for comparison and what Riley asked for.
    """
    df = pd.read_csv(grades_csv_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bins = np.arange(0, 101, 5)
    courses = ["MATH100", "STAT200", "COMM293"]

    plt.figure(figsize=(10, 8))
    for i, course in enumerate(courses, start=1):
        plt.subplot(3, 1, i)
        subset = df.loc[df["course"] == course, "grade"]
        plt.hist(subset, bins=bins)
        plt.title(course)
        plt.xlabel("Grade")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    os.makedirs("data", exist_ok=True)

    generate_weather_dataset("data/weather_station_dirty_timestamps.csv")
    generate_coffee_dataset("data/coffee_eng_vs_sci.csv")
    generate_grades_dataset("data/grades_three_courses.csv")

    # Figures (useful previews for each dataset)
    os.makedirs("figures", exist_ok=True)

    save_weather_preview(
        "data/weather_station_dirty_timestamps.csv",
        "figures/weather_temp_over_time.png"
    )

    save_coffee_preview(
        "data/coffee_eng_vs_sci.csv",
        "figures/coffee_by_faculty_boxplot.png"
    )

    # Grades: both overlay and faceted histograms
    save_grades_overlay_histogram(
        "data/grades_three_courses.csv",
        "figures/grades_histogram_overlay.png"
    )

    save_grades_faceted_histograms(
        "data/grades_three_courses.csv",
        "figures/grades_histogram_faceted.png"
    )

    print("- figures/weather_temp_over_time.png")
    print("- figures/coffee_by_faculty_boxplot.png")
    print("- figures/grades_histogram_overlay.png")
    print("- figures/grades_histogram_faceted.png")

    




if __name__ == "__main__":
    main()
