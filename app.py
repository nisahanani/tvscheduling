# app.py
import streamlit as st
import pandas as pd
import random
from typing import List, Tuple, Dict
from pathlib import Path

# =========================
# Data loading & utilities
# =========================
def load_ratings(csv_path_or_file) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Loads ratings CSV with columns:
        "Type of Program", "Hour 6", "Hour 7", ... "Hour 23"
    Accepts a path string or an UploadedFile/file-like object.
    Returns (df, programs, hour_cols)
    """
    df = pd.read_csv(csv_path_or_file)
    if "Type of Program" not in df.columns:
        raise ValueError("CSV must include a 'Type of Program' column.")
    # Hour columns are all except the first
    hour_cols = [c for c in df.columns if c != "Type of Program"]
    if len(hour_cols) == 0:
        raise ValueError("CSV must include hour columns (e.g. 'Hour 6', 'Hour 7', ...).")
    programs = df["Type of Program"].astype(str).tolist()
    return df, programs, hour_cols

def fitness(schedule: List[str], df: pd.DataFrame, hour_cols: List[str]) -> float:
    """
    Sum of ratings for chosen program at each hour.
    schedule length must equal len(hour_cols).
    """
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"].astype(str))}
    total = 0.0
    for idx, program in enumerate(schedule):
        row_i = program_to_row[program]
        hour_col = hour_cols[idx]
        total += float(df.at[row_i, hour_col])
    return total

def random_schedule(programs: List[str], num_hours: int) -> List[str]:
    # Allow repeats — a program can appear in multiple hours
    return [random.choice(programs) for _ in range(num_hours)]

def single_point_crossover(p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
    if len(p1) < 2:
        return p1[:], p2[:]
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

def mutate(schedule: List[str], programs: List[str]) -> List[str]:
    i = random.randrange(len(schedule))
    schedule[i] = random.choice(programs)
    return schedule

def tournament_selection(pop: List[List[str]], df: pd.DataFrame, hour_cols: List[str], k: int = 3) -> List[str]:
    candidates = random.sample(pop, k=min(k, len(pop)))
    candidates.sort(key=lambda s: fitness(s, df, hour_cols), reverse=True)
    return candidates[0]

def run_ga(
    df: pd.DataFrame,
    programs: List[str],
    hour_cols: List[str],
    generations: int = 100,
    pop_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism: int = 2,
    tournament_k: int = 3,
) -> Dict:
    num_hours = len(hour_cols)
    # Initialize population of random schedules (length = number of hour slots)
    population = [random_schedule(programs, num_hours) for _ in range(pop_size)]
    best = max(population, key=lambda s: fitness(s, df, hour_cols))
    best_score = fitness(best, df, hour_cols)

    for _ in range(generations):
        new_pop = []
        # Elitism: keep the top 'elitism' individuals
        population.sort(key=lambda s: fitness(s, df, hour_cols), reverse=True)
        new_pop.extend(population[:elitism])

        while len(new_pop) < pop_size:
            # Parent selection via tournament
            p1 = tournament_selection(population, df, hour_cols, k=tournament_k)
            p2 = tournament_selection(population, df, hour_cols, k=tournament_k)

            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            # Mutation
            if random.random() < mutation_rate:
                c1 = mutate(c1, programs)
            if random.random() < mutation_rate:
                c2 = mutate(c2, programs)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        # Track global best
        gen_best = max(population, key=lambda s: fitness(s, df, hour_cols))
        gen_score = fitness(gen_best, df, hour_cols)
        if gen_score > best_score:
            best, best_score = gen_best, gen_score

    return {
        "best_schedule": best,
        "best_score": best_score,
    }

def render_schedule_table(schedule: List[str], df: pd.DataFrame, hour_cols: List[str]) -> pd.DataFrame:
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"].astype(str))}
    rows = []
    for idx, program in enumerate(schedule):
        hour_label = hour_cols[idx]
        # extract HH from "Hour 6" etc
        hour_number = hour_label.split()[-1]
        rating = float(df.at[program_to_row[program], hour_label])
        rows.append({
            "Hour": f"{hour_number}:00",
            "Program": program,
            "Rating": rating
        })
    return pd.DataFrame(rows)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="GA TV Scheduling")
st.title(" Genetic Algorithm — TV Scheduling")

# Use __file__ (two underscores) — with fallback to cwd if __file__ is not defined.
try:
    default_csv = Path(__file__).parent / "program_ratings.csv"
except NameError:
    # __file__ may not exist in some interactive environments; use current working dir instead
    default_csv = Path.cwd() / "program_ratings.csv"

if uploaded is not None:
    try:
        df, programs, hour_cols = load_ratings(uploaded)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
elif default_csv.exists():
    try:
        df, programs, hour_cols = load_ratings(str(default_csv))
    except Exception as e:
        st.error(f"Error reading default CSV ({default_csv}): {e}")
        st.stop()
else:
    st.error("No CSV found. Upload a file or add program_ratings.csv to the same folder as app.py.")
    st.stop()
# ---------------------------------------------------------------

st.subheader("Parameters")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("*Trial 1*")
    co1 = st.slider("CO_R (Trial 1)", 0.0, 0.95, 0.80, 0.01, key="co1")
    mu1 = st.slider("MUT_R (Trial 1)", 0.01, 0.05, 0.02, 0.01, key="mu1")

    st.markdown("*Trial 2*")
    co2 = st.slider("CO_R (Trial 2)", 0.0, 0.95, 0.70, 0.01, key="co2")
    mu2 = st.slider("MUT_R (Trial 2)", 0.01, 0.05, 0.03, 0.01, key="mu2")

with col_b:
    st.markdown("*Trial 3*")
    co3 = st.slider("CO_R (Trial 3)", 0.0, 0.95, 0.60, 0.01, key="co3")
    mu3 = st.slider("MUT_R (Trial 3)", 0.01, 0.05, 0.04, 0.01, key="mu3")

st.markdown("---")
st.subheader("Global GA Settings")
gen = st.number_input("Generations (GEN)", min_value=10, max_value=2000, value=100, step=10)
pop = st.number_input("Population Size (POP)", min_value=10, max_value=500, value=50, step=10)
elit = st.number_input("Elitism Size", min_value=0, max_value=10, value=2, step=1)
tourn = st.number_input("Tournament Size (k)", min_value=2, max_value=10, value=3, step=1)

if st.button("Run All 3 Trials 🚀", use_container_width=True):
    trials = [
        ("Trial 1", co1, mu1),
        ("Trial 2", co2, mu2),
        ("Trial 3", co3, mu3),
    ]

    for label, co, mu in trials:
        st.markdown(f"### {label}")
        with st.spinner(f"Running {label} (CO_R={co:.2f}, MUT_R={mu:.2f})..."):
            result = run_ga(
                df, programs, hour_cols,
                generations=int(gen),
                pop_size=int(pop),
                crossover_rate=float(co),
                mutation_rate=float(mu),
                elitism=int(elit),
                tournament_k=int(tourn),
            )
        schedule = result["best_schedule"]
        score = result["best_score"]
        table = render_schedule_table(schedule, df, hour_cols)
        st.dataframe(table, use_container_width=True)
        st.metric(label="Total Ratings (Fitness)", value=round(score, 4))
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(label=f"Download {label} CSV", data=csv_bytes, file_name=f"{label.replace(' ', '_')}_schedule.csv", mime="text/csv")
        st.caption(f"CO_R = {co:.2f}, MUT_R = {mu:.2f}, GEN = {gen}, POP = {pop}, ELIT = {elit}, TOURN = {tourn}")
        st.markdown("---")
