import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
import random

# =========================
# Data loading
# =========================
@st.cache_data
def load_ratings(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Loads ratings CSV with columns:
        "Type of Program", "Hour 6", "Hour 7", ... "Hour 23"
    Returns (df, programs, hour_cols)
    """
    df = pd.read_csv(csv_path)
    if "Type of Program" not in df.columns:
        raise ValueError("CSV must include a 'Type of Program' column.")
    hour_cols = [c for c in df.columns if c != "Type of Program"]
    programs = df["Type of Program"].tolist()
    return df, programs, hour_cols

def fitness(schedule: List[str], df: pd.DataFrame, hour_cols: List[str]) -> float:
    """
    Sum of ratings for chosen program at each hour.
    schedule length must equal len(hour_cols).
    """
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"])}
    total = 0.0
    for idx, program in enumerate(schedule):
        row_i = program_to_row[program]
        hour_col = hour_cols[idx]
        total += float(df.at[row_i, hour_col])
    return total

# ---------- RNG-aware operators ----------
def random_schedule(programs: List[str], num_hours: int, rng: random.Random) -> List[str]:
    # Allow repeats — a program can appear in multiple hours
    return [rng.choice(programs) for _ in range(num_hours)]

def single_point_crossover(p1: List[str], p2: List[str], rng: random.Random) -> Tuple[List[str], List[str]]:
    if len(p1) < 2:
        return p1[:], p2[:]
    cut = rng.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

def mutate(schedule: List[str], programs: List[str], rng: random.Random) -> List[str]:
    # Mutate exactly one gene position
    i = rng.randrange(len(schedule))
    schedule = schedule[:]  # copy before mutate
    schedule[i] = rng.choice(programs)
    return schedule

def tournament_selection(pop: List[List[str]], df: pd.DataFrame, hour_cols: List[str], rng: random.Random, k: int = 3) -> List[str]:
    # Sample k candidates without replacement (cap at len(pop))
    k = min(k, len(pop))
    idxs = rng.sample(range(len(pop)), k)
    candidates = [pop[i] for i in idxs]
    # Stable sort: tie-break by genotype to keep deterministic order
    candidates.sort(key=lambda s: (fitness(s, df, hour_cols), tuple(s)), reverse=True)
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
    seed: int = 42,  # <--- NEW
) -> Dict:
    rng = random.Random(seed)  # local RNG, reproducible per run
    num_hours = len(hour_cols)

    # Initialize population
    population = [random_schedule(programs, num_hours, rng) for _ in range(pop_size)]

    # Track best
    population.sort(key=lambda s: (fitness(s, df, hour_cols), tuple(s)), reverse=True)
    best = population[0][:]
    best_score = fitness(best, df, hour_cols)

    for _ in range(generations):
        new_pop = []
        # Elitism (copy to avoid aliasing)
        elites = [ind[:] for ind in population[:elitism]]
        new_pop.extend(elites)

        # Fill the rest
        while len(new_pop) < pop_size:
            # Parent selection
            p1 = tournament_selection(population, df, hour_cols, rng, k=tournament_k)
            p2 = tournament_selection(population, df, hour_cols, rng, k=tournament_k)

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            # Mutation (per-child probability, same as your original intent)
            if rng.random() < mutation_rate:
                c1 = mutate(c1, programs, rng)
            if rng.random() < mutation_rate:
                c2 = mutate(c2, programs, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # Prepare next generation
        population = new_pop
        population.sort(key=lambda s: (fitness(s, df, hour_cols), tuple(s)), reverse=True)

        # Update global best
        if (fs := fitness(population[0], df, hour_cols)) > best_score:
            best_score = fs
            best = population[0][:]

    return {"best_schedule": best, "best_score": best_score}

def render_schedule_table(schedule: List[str], df: pd.DataFrame, hour_cols: List[str]) -> pd.DataFrame:
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"])}
    rows = []
    for idx, program in enumerate(schedule):
        hour_label = hour_cols[idx]
        hour_number = hour_label.split()[-1]  # from "Hour 6" -> "6"
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
st.set_page_config(page_title="GA TV Scheduling", layout="wide")
st.title("Genetic Algorithm — TV Scheduling")

# ---- CSV loading: _file_ may not exist on Streamlit Cloud, fall back to uploader ----
default_csv = None
try:
    # _file_ exists in local runs; not guaranteed on every platform
    here = Path(_file_).parent
    candidate = here / "program_ratings.csv"
    if candidate.exists():
        default_csv = str(candidate)
except NameError:
    pass

csv_path = None
if default_csv:
    csv_path = default_csv
else:
    st.info("Upload your ratings CSV (must include a 'Type of Program' column).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        # Save to tmp file for caching to work
        tmp = Path("uploaded_program_ratings.csv")
        tmp.write_bytes(up.read())
        csv_path = str(tmp)

if not csv_path:
    st.error("No CSV provided. Please place 'program_ratings.csv' next to the app or upload it above.")
    st.stop()

df, programs, hour_cols = load_ratings(csv_path)

# ------------------- Trials & Seeds -------------------
st.subheader("Parameters")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("*Trial 1*")
    co1 = st.slider("CO_R (Trial 1)", 0.0, 0.95, 0.80, 0.01, key="co1")
    mu1 = st.slider("MUT_R (Trial 1)", 0.01, 0.05, 0.02, 0.01, key="mu1")
    s1  = st.number_input("Seed (Trial 1)", value=123, step=1, key="s1")

with col_b:
    st.markdown("*Trial 2*")
    co2 = st.slider("CO_R (Trial 2)", 0.0, 0.95, 0.70, 0.01, key="co2")
    mu2 = st.slider("MUT_R (Trial 2)", 0.01, 0.05, 0.03, 0.01, key="mu2")
    s2  = st.number_input("Seed (Trial 2)", value=456, step=1, key="s2")

with col_c:
    st.markdown("*Trial 3*")
    co3 = st.slider("CO_R (Trial 3)", 0.0, 0.95, 0.60, 0.01, key="co3")
    mu3 = st.slider("MUT_R (Trial 3)", 0.01, 0.05, 0.04, 0.01, key="mu3")
    s3  = st.number_input("Seed (Trial 3)", value=789, step=1, key="s3")

st.markdown("---")
st.subheader("Global GA Settings")
colg1, colg2, colg3, colg4 = st.columns(4)
with colg1:
    gen = st.number_input("Generations (GEN)", min_value=10, max_value=2000, value=100, step=10)
with colg2:
    pop = st.number_input("Population Size (POP)", min_value=10, max_value=500, value=50, step=10)
with colg3:
    elit = st.number_input("Elitism Size", min_value=0, max_value=10, value=2, step=1)
with colg4:
    tourn = st.number_input("Tournament Size (k)", min_value=2, max_value=10, value=3, step=1)

if st.button("Run All 3 Trials", use_container_width=True):
    trials = [
        ("Trial 1", co1, mu1, s1),
        ("Trial 2", co2, mu2, s2),
        ("Trial 3", co3, mu3, s3),
    ]

    for label, co, mu, seed in trials:
        st.markdown(f"### {label}")
        result = run_ga(
            df, programs, hour_cols,
            generations=int(gen),
            pop_size=int(pop),
            crossover_rate=float(co),
            mutation_rate=float(mu),
            elitism=int(elit),
            tournament_k=int(tourn),
            seed=int(seed),  # <--- reproducible per trial
        )
        schedule = result["best_schedule"]
        score = result["best_score"]
        table = render_schedule_table(schedule, df, hour_cols)
        st.dataframe(table, use_container_width=True)
        st.metric(label="Total Ratings (Fitness)", value=round(score, 4))
        st.caption(
            f"CO_R = {co:.2f}, MUT_R = {mu:.2f}, GEN = {gen}, POP = {pop}, ELIT = {elit}, TOURN = {tourn}, SEED = {seed}"
        )
        st.markdown("---")
