"""TidyTuesday 2026-04-07: Repair Cafes Worldwide.

Answers the four guiding questions from the readme and writes charts to figures/.
Usage: python analysis.py [data_dir]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlelocation": "left", "axes.titlesize": 13,
    "axes.titleweight": "bold", "font.size": 10, "figure.dpi": 120,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)


# --- Load & clean ------------------------------------------------------------
repairs = pd.read_csv(DATA / "repairs.csv", parse_dates=["repair_date"])
text = pd.read_csv(DATA / "repairs_text.csv", low_memory=False)

# The two files are row-aligned splits of one sheet. repair_id is not unique:
# 33 IDs are shared by two unrelated items (ID collisions, not repeat visits),
# so join by position rather than on repair_id, and keep every row.
assert repairs["repair_id"].equals(text["repair_id"]), "files are not row-aligned"
df = pd.concat([repairs, text.drop(columns="repair_id")], axis=1)
df["row_key"] = df["repair_id"] + "#" + df.index.astype(str)
df["repaired"] = df["repaired"].replace({"ja": "yes"})
df = df[df["repaired"].isin(["yes", "half", "no"])]
# Data was published 2026-04-07; a handful of rows are dated after that.
df = df[df["repair_date"] <= "2026-04-07"]
df["year"] = df["repair_date"].dt.year
df["success"] = df["repaired"].eq("yes")
print(f"{len(df):,} repairs, {df.repair_cafe_number.nunique()} cafes, "
      f"{df.country.nunique()} countries, {df.repair_date.min():%Y-%m} to "
      f"{df.repair_date.max():%Y-%m}")
print(df["repaired"].value_counts(normalize=True).round(3).to_dict())

# --- Q1: What kinds of items are most easily repaired? ------------------------
cat = (df.groupby("category")
         .agg(n=("success", "size"), success=("success", "mean"),
              repairability=("repairability", "mean"))
         .sort_values("success"))
print("\nRepair success by category\n", cat.round(3))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(cat.index, cat["success"], color=BLUE, height=0.7)
for y, (s, n) in enumerate(zip(cat["success"], cat["n"])):
    ax.text(s + 0.01, y, f"{s:.0%}  (n={n:,})", va="center", color=INK2, fontsize=9)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Share of items fully repaired, by category")
save(fig, "q1_success_by_category.png")

prod = (df.groupby("kind_of_product")
          .agg(n=("success", "size"), success=("success", "mean"))
          .query("n >= 500").sort_values("success"))
ends = pd.concat([prod.head(8), prod.tail(8)])
print("\nEasiest / hardest products (n >= 500)\n", ends.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
colors = [ORANGE] * 8 + [BLUE] * 8
ax.barh(ends.index, ends["success"], color=colors, height=0.7)
for y, s in enumerate(ends["success"]):
    ax.text(s + 0.01, y, f"{s:.0%}", va="center", color=INK2, fontsize=9)
ax.axhline(7.5, color=GRAY, lw=0.8, ls="--")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Hardest (orange) and easiest (blue) products to fix\n"
             "products with at least 500 repair attempts")
save(fig, "q1_products_extremes.png")

# --- Q2: Most common reasons items can't be repaired --------------------------
failed = df[df["repaired"] == "no"]
reasons = failed["failure_reasons"].value_counts()
coverage = failed["failure_reasons"].notna().mean()
print(f"\nFailure reasons (recorded for {coverage:.0%} of failed repairs)\n",
      (reasons / reasons.sum()).round(3))

fig, ax = plt.subplots(figsize=(8, 4.5))
r = reasons.sort_values()
ax.barh(r.index, r.values, color=BLUE, height=0.7)
for y, v in enumerate(r.values):
    ax.text(v + 50, y, f"{v:,}  ({v / r.sum():.0%})", va="center", color=INK2, fontsize=9)
ax.set_xlim(0, r.max() * 1.25)
ax.set_title(f"Why repairs failed\n{r.sum():,} failed repairs with a recorded reason")
save(fig, "q2_failure_reasons.png")

top_cats = df["category"].value_counts().head(6).index
mix = (failed[failed["category"].isin(top_cats)]
       .groupby("category")["failure_reasons"].value_counts(normalize=True)
       .unstack().fillna(0))
print("\nFailure-reason mix within top categories\n", mix.round(2).T)

# --- Q3: Which countries have seen the most growth in Repair Cafe branches? ---
# "Active" = a branch that logged at least one repair that year. 2026 is a
# partial year so growth is measured 2022 -> 2025.
active = df.groupby(["country", "year"])["repair_cafe_number"].nunique().unstack(fill_value=0)
first_seen = df.groupby("repair_cafe_number").agg(country=("country", "first"),
                                                  first=("year", "min"))
new_since = (first_seen.query("first >= 2023").groupby("country").size()
             .rename("new_branches_2023_25"))
growth = active[[2019, 2022, 2025]].join(new_since).fillna(0).astype(int)
growth["change_22_25"] = growth[2025] - growth[2022]
growth = growth.sort_values("change_22_25", ascending=False)
print("\nActive branches per country\n", growth.head(12))

top = growth.head(6).index
fig, ax = plt.subplots(figsize=(8, 4.5))
yrs = [y for y in active.columns if y <= 2025]
palette = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300"]
for c, col in zip(top, palette):
    ax.plot(yrs, active.loc[c, yrs], color=col, lw=2, marker="o", ms=4)
    ax.text(yrs[-1] + 0.2, active.loc[c, 2025], c, color=INK, va="center", fontsize=9)
ax.set_xticks(yrs)
ax.set_ylabel("Branches logging repairs")
ax.set_title("Active Repair Cafe branches per year (top 6 by growth, 2022-25)")
save(fig, "q3_branch_growth.png")

# --- Q4: Is GenAI overtaking YouTube as a repair info source? -----------------
# Source and URL fields are free text and were often swapped by submitters,
# so search both. Keyword lists are deliberately explicit: a bare "ai" would
# match French "j'ai".
src = (df["repair_info_source"].fillna("") + " " +
       df["repair_info_url"].fillna("")).str.lower()
patterns = {
    "YouTube": r"you ?tube|\byt\b|youtu\.be",
    "GenAI": r"chat ?gpt|\bgpt\b|openai|gemini|copilot|perplexity|claude|\bllm\b|\bai\b",
    "iFixit": r"ifixit",
}
named = src.str.strip().ne("") & ~src.str.strip().isin(["supplier", "user"])
q4 = pd.DataFrame({k: src.str.contains(p) for k, p in patterns.items()})
q4["year"] = df["year"]
by_year = q4.groupby("year").sum()
by_year["records_naming_a_source"] = named.groupby(df["year"]).sum()
print("\nInfo-source mentions per year\n", by_year)

fig, ax = plt.subplots(figsize=(8, 4.5))
yrs = [y for y in by_year.index if 2017 <= y <= 2025]
for k, col, nudge in zip(["YouTube", "iFixit", "GenAI"], [BLUE, AQUA, ORANGE], [0, 8, -8]):
    ax.plot(yrs, by_year.loc[yrs, k], color=col, lw=2, marker="o", ms=4)
    ax.text(yrs[-1] + 0.2, by_year.loc[yrs[-1], k] + nudge, k, color=INK, va="center", fontsize=9)
ax.set_xticks(yrs)
ax.set_ylabel("Repairs citing source")
ax.set_title("YouTube still dwarfs GenAI as a cited repair source")
save(fig, "q4_info_sources.png")

print(f"\nFigures written to {FIGS.resolve()}")
