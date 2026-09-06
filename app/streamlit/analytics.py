"""Pure pandas/number helpers for the dashboard.

Nothing here imports ``streamlit`` — every function takes plain DataFrames and
returns plain DataFrames / lists, so it can be unit-tested without a runtime.
Caching and I/O live in ``data.py``; figure construction in ``charts.py``.
"""

from __future__ import annotations

import math

import pandas as pd

# ── formatting ──────────────────────────────────────────────────────────────


def format_color(color: str | None) -> str:
    """Normalise a hex colour to lower-case ``#rrggbb`` form.

    Missing / non-string / ``"nan"`` values (historical seasons have no team
    colour) fall back to white rather than producing an invalid ``"#nan"``.
    """
    if color is None or not isinstance(color, str):
        return "#ffffff"
    s = color.strip()
    if not s or s.lower() == "nan":
        return "#ffffff"
    if s.startswith("#"):
        return s.lower()
    return f"#{s}".lower()


def short_name(full_name: str) -> str:
    """``"Max Verstappen" -> "M. Verstappen"`` (leaves single tokens alone)."""
    parts = str(full_name).split()
    if len(parts) < 2:
        return str(full_name)
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def driver_label(full_name: str, team_name: str) -> str:
    """Human label for filter widgets: ``"M. Verstappen — Red Bull Racing"``."""
    team = f" — {team_name}" if team_name else ""
    return f"{short_name(full_name)}{team}"


def _rank_by(df: pd.DataFrame, col: str = "Points") -> pd.DataFrame:
    """Sort descending by ``col`` and insert a 1-based ``Rank`` column."""
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def _color_map(
    df: pd.DataFrame, key_col: str, color_col: str = "TeamColor", keep: str = "last"
) -> dict:
    """Build a ``{key: color}`` lookup, one entry per unique ``key_col`` value."""
    return (
        df.drop_duplicates(subset=key_col, keep=keep)[[key_col, color_col]]
        .set_index(key_col)[color_col]
        .to_dict()
    )


# ── season aggregates (from Bronze race rows) ───────────────────────────────


def _season_races(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Race-mode rows for one season with numeric position/grid/points + DNF."""
    df = races.copy()
    if "Mode" in df.columns:
        df = df[df["Mode"] == "Race"]
    df = df[df["Year"] == year].copy()
    if df.empty:
        return df
    for col in ("Position", "GridPosition", "Points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "DNF" not in df.columns:
        df["DNF"] = df.get("ClassifiedPosition", pd.Series(index=df.index)) == "R"
    return df


def compute_driver_stats(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-driver season aggregates: wins, podiums, poles, DNFs, pace metrics."""
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()

    stats = df.groupby(
        ["FullName", "TeamName", "TeamColor", "Abbreviation"], as_index=False
    ).agg(
        Races=("RoundNumber", "nunique"),
        Points=("Points", "sum"),
        Wins=("Position", lambda s: int((s == 1).sum())),
        Podiums=("Position", lambda s: int((s <= 3).sum())),
        Poles=("GridPosition", lambda s: int((s == 1).sum())),
        DNFs=("DNF", "sum"),
        AvgFinish=("Position", "mean"),
        AvgGrid=("GridPosition", "mean"),
        BestFinish=("Position", "min"),
    )
    stats["AvgGain"] = stats["AvgGrid"] - stats["AvgFinish"]
    stats["PodiumRate"] = stats["Podiums"] / stats["Races"]
    return _rank_by(stats)


def compute_team_stats(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-constructor season aggregates."""
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()

    stats = df.groupby(["TeamName", "TeamColor"], as_index=False).agg(
        Points=("Points", "sum"),
        Wins=("Position", lambda s: int((s == 1).sum())),
        Podiums=("Position", lambda s: int((s <= 3).sum())),
    )
    return _rank_by(stats)


def compute_reliability(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-driver finishing reliability & grid-vs-race movement for a season."""
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()

    df["Gain"] = df["GridPosition"] - df["Position"]
    out = df.groupby(
        ["FullName", "Abbreviation", "TeamName", "TeamColor"], as_index=False
    ).agg(
        Starts=("RoundNumber", "nunique"),
        DNFs=("DNF", "sum"),
        Points=("Points", "sum"),
        PointsFinishes=("Points", lambda s: int((s > 0).sum())),
        AvgGrid=("GridPosition", "mean"),
        BestGrid=("GridPosition", "min"),
        AvgGain=("Gain", "mean"),
        GainStd=("Gain", "std"),
    )
    out["DNFRate"] = out["DNFs"] / out["Starts"]
    out["PointsFinishRate"] = out["PointsFinishes"] / out["Starts"]
    return out.sort_values("Points", ascending=False).reset_index(drop=True)


def compute_teammate_h2h(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Intra-constructor duel: race / qualifying wins and points split.

    Only rounds where a constructor fielded exactly two drivers count. Driver
    ``a`` is the higher season points scorer (alphabetical tie-break).
    """
    df = _season_races(races, year)
    if df.empty or "TeamId" not in df.columns:
        return pd.DataFrame()

    rows = []
    for team_id, team_df in df.groupby("TeamId"):
        team_name = team_df["TeamName"].iloc[-1]
        team_color = team_df["TeamColor"].iloc[-1]

        points = team_df.groupby("DriverId")["Points"].sum()
        drivers = sorted(points.index, key=lambda d: (-points[d], d))
        if len(drivers) != 2:
            continue
        a, b = drivers
        abbr = team_df.drop_duplicates("DriverId").set_index("DriverId")["Abbreviation"]

        rec = {
            "TeamId": team_id,
            "TeamName": team_name,
            "TeamColor": team_color,
            "driver_a": a,
            "driver_b": b,
            "abbr_a": abbr.get(a, a),
            "abbr_b": abbr.get(b, b),
            "Rounds": 0,
            "RaceWinsA": 0,
            "RaceWinsB": 0,
            "QualiWinsA": 0,
            "QualiWinsB": 0,
            "PointsA": float(points[a]),
            "PointsB": float(points[b]),
        }

        for _, rnd in team_df.groupby("RoundNumber"):
            pos = rnd.set_index("DriverId")["Position"]
            grid = rnd.set_index("DriverId")["GridPosition"]
            if a not in pos.index or b not in pos.index:
                continue
            rec["Rounds"] += 1
            if pd.notna(pos[a]) or pd.notna(pos[b]):
                pa = pos[a] if pd.notna(pos[a]) else math.inf
                pb = pos[b] if pd.notna(pos[b]) else math.inf
                if pa < pb:
                    rec["RaceWinsA"] += 1
                elif pb < pa:
                    rec["RaceWinsB"] += 1
            if pd.notna(grid[a]) and pd.notna(grid[b]):
                if grid[a] < grid[b]:
                    rec["QualiWinsA"] += 1
                elif grid[b] < grid[a]:
                    rec["QualiWinsB"] += 1

        rows.append(rec)

    return (
        pd.DataFrame(rows)
        .sort_values("PointsA", ascending=False)
        .reset_index(drop=True)
    )


# ── prediction momentum (from the ABT + /predict probabilities) ─────────────


def _slope(y) -> float:
    """Least-squares slope of ``y`` against 0..n-1; 0.0 for < 2 points."""
    ys = [float(v) for v in y if pd.notna(v)]
    n = len(ys)
    if n < 2:
        return 0.0
    xs = range(n)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (v - my) for x, v in zip(xs, ys)) / den


def compute_momentum(
    preds: pd.DataFrame, last_n: int = 5, flat_eps: float = 0.005
) -> pd.DataFrame:
    """Per-driver win-probability trajectory: latest value, delta vs previous
    prediction date, slope over the last ``last_n`` dates, and rank change."""
    if preds.empty:
        return pd.DataFrame()

    meta_cols = [
        c
        for c in ("FullName", "Abbreviation", "TeamName", "TeamColor", "HeadshotUrl")
        if c in preds.columns
    ]
    dates = sorted(preds["dt_ref"].unique())
    latest_date = dates[-1]
    prev_date = dates[-2] if len(dates) >= 2 else None

    rows = []
    for driver_id, g in preds.sort_values("dt_ref").groupby("DriverId"):
        series = g.set_index("dt_ref")["prob_win"]
        latest = series.get(latest_date, float("nan"))
        prev = (
            series.get(prev_date, float("nan"))
            if prev_date is not None
            else float("nan")
        )
        tail = series.tail(last_n).tolist()
        slope = _slope(tail)
        meta = {c: g[c].iloc[-1] for c in meta_cols}
        rows.append(
            {
                "DriverId": driver_id,
                **meta,
                "latest": latest,
                "prev": prev,
                "delta_prev": latest - prev,
                "slope": slope,
                "trend": "up"
                if slope > flat_eps
                else "down"
                if slope < -flat_eps
                else "flat",
            }
        )

    out = pd.DataFrame(rows)
    out["rank"] = out["latest"].rank(ascending=False, method="min")
    out["prev_rank"] = out["prev"].rank(ascending=False, method="min")
    out["rank_change"] = out["prev_rank"] - out["rank"]
    return out.sort_values("latest", ascending=False).reset_index(drop=True)


def top_factors(
    importances: dict[str, float],
    feature_row: pd.Series,
    field_median: pd.Series,
    k: int = 6,
) -> pd.DataFrame:
    """Heuristic 'why' for one driver: the highest-importance features, each
    flagged by whether this driver sits above or below the field median.

    Not SHAP — it multiplies global RF importance by the sign of the driver's
    deviation from the median to give a rough push direction.
    """
    if not importances:
        return pd.DataFrame()

    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    rows = []
    for feat, imp in ranked:
        if feat not in feature_row.index:
            continue
        val = feature_row[feat]
        med = field_median.get(feat, float("nan"))
        if pd.isna(val) or pd.isna(med):
            direction = "—"
        elif val > med:
            direction = "above field"
        elif val < med:
            direction = "below field"
        else:
            direction = "at field"
        rows.append(
            {
                "feature": feat,
                "importance": imp,
                "value": val,
                "field_median": med,
                "vs_field": direction,
            }
        )
        if len(rows) == k:
            break
    return pd.DataFrame(rows)


# ── narrative insights ─────────────────────────────────────────────────────


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def build_insights(
    momentum: pd.DataFrame,
    h2h: pd.DataFrame,
    reliability: pd.DataFrame,
    last_n: int = 5,
) -> list[str]:
    """Up to five deterministic one-line findings for the header panel."""
    out: list[str] = []

    if not momentum.empty and momentum["delta_prev"].notna().any():
        m = momentum.loc[momentum["delta_prev"].abs().idxmax()]
        name = m.get("FullName", m["DriverId"])
        out.append(
            f"📈 **{name}** is the biggest mover since the last round "
            f"({_pp(m['delta_prev'])} win probability, now {m['latest']:.0%})."
        )

    if not momentum.empty and (momentum["slope"] > 0).any():
        s = momentum.sort_values("slope", ascending=False).iloc[0]
        name = s.get("FullName", s["DriverId"])
        out.append(
            f"🚀 **{name}** has the fastest-rising form "
            f"(+{s['slope'] * 100:.1f} pp/round over the last {last_n})."
        )

    if not h2h.empty:
        close = h2h.assign(margin=(h2h["RaceWinsA"] - h2h["RaceWinsB"]).abs())
        close = close[close["Rounds"] >= 3].sort_values(
            ["margin", "Rounds"], ascending=[True, False]
        )
        if not close.empty:
            c = close.iloc[0]
            out.append(
                f"⚔️ Closest teammate fight: **{c['abbr_a']} {int(c['RaceWinsA'])}–"
                f"{int(c['RaceWinsB'])} {c['abbr_b']}** on race day at {c['TeamName']}."
            )

    if not reliability.empty:
        rel = reliability[reliability["Starts"] >= 3]
        if not rel.empty:
            q = rel.sort_values("AvgGrid").iloc[0]
            out.append(
                f"🎯 **{q['FullName']}** is the strongest qualifier "
                f"(avg grid P{q['AvgGrid']:.1f})."
            )
            top10 = rel.head(10)
            worst = top10.sort_values("DNFRate", ascending=False).iloc[0]
            if worst["DNFs"] > 0:
                out.append(
                    f"🔧 Reliability watch: **{worst['FullName']}** has retired from "
                    f"{int(worst['DNFs'])}/{int(worst['Starts'])} races "
                    f"({worst['DNFRate']:.0%})."
                )

    return out[:5]
