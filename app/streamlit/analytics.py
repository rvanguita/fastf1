"""Camada semântica do produto analítico de Fórmula 1.

Funções puras: I/O e cache ficam em ``data.py``; Plotly fica em ``charts.py``.
"""

from __future__ import annotations

import math
import re

import pandas as pd

STATUS_LABELS = {"R": "DNF", "D": "DSQ", "E": "DSQ", "W": "DNS", "F": "DNQ", "N": "NC"}


def format_color(color: str | None) -> str:
    """Normaliza cores FastF1; ausentes usam cinza acessível."""
    if color is None or not isinstance(color, str):
        return "#737b8c"
    value = color.strip()
    if not value or value.lower() == "nan":
        return "#737b8c"
    return value.lower() if value.startswith("#") else f"#{value}".lower()


def short_name(full_name: str) -> str:
    parts = str(full_name).split()
    return str(full_name) if len(parts) < 2 else f"{parts[0][0]}. {' '.join(parts[1:])}"


def driver_label(full_name: str, team_name: str) -> str:
    return (
        f"{short_name(full_name)} — {team_name}" if team_name else short_name(full_name)
    )


def _rank_by(df: pd.DataFrame, col: str = "Points") -> pd.DataFrame:
    sort_cols = [col, "FullName"] if "FullName" in df else [col]
    ascending = [False, True] if "FullName" in df else False
    out = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out


def _color_map(
    df: pd.DataFrame, key_col: str, color_col: str = "TeamColor", keep: str = "last"
) -> dict:
    return (
        df.drop_duplicates(subset=key_col, keep=keep)[[key_col, color_col]]
        .set_index(key_col)[color_col]
        .map(format_color)
        .to_dict()
    )


def classify_result(classified_position, status: str | None = None) -> str:
    """Converte a classificação FastF1 em uma categoria analítica explícita."""
    value = str(classified_position).strip().upper()
    if re.fullmatch(r"\d+(\.0)?", value):
        return "FINISHED"
    if value in STATUS_LABELS:
        return STATUS_LABELS[value]
    status_value = str(status or "").lower()
    if "did not start" in status_value or "withdraw" in status_value:
        return "DNS"
    if "disqual" in status_value or "excluded" in status_value:
        return "DSQ"
    if status_value == "finished" or status_value.startswith("+"):
        return "FINISHED"
    return "DNF"


def _season_races(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Resultados de corrida com tipos e semântica consistentes."""
    df = races.copy()
    if "Mode" in df.columns:
        df = df[df["Mode"] == "Race"]
    df = df[df["Year"] == year].copy()
    if df.empty:
        return df
    for col in ("Position", "GridPosition", "Points", "RoundNumber"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    classified = df.get("ClassifiedPosition", df["Position"])
    df["OfficialFinish"] = pd.to_numeric(classified, errors="coerce")
    df["OfficialGrid"] = df["GridPosition"].where(df["GridPosition"] > 0)
    statuses = df.get("Status", pd.Series("", index=df.index))
    df["ResultStatus"] = [
        classify_result(cp, st) for cp, st in zip(classified, statuses)
    ]
    df["Started"] = ~df["ResultStatus"].isin(["DNS", "DNQ"])
    df["DNF"] = df["ResultStatus"] == "DNF"
    df["Gain"] = df["OfficialGrid"] - df["OfficialFinish"]
    return df


def _latest_metadata(
    df: pd.DataFrame, keys: list[str], fields: tuple[str, ...] | None = None
) -> pd.DataFrame:
    candidates = fields or (
        "FullName",
        "Abbreviation",
        "TeamId",
        "TeamName",
        "TeamColor",
        "HeadshotUrl",
    )
    meta = [c for c in candidates if c in df.columns and c not in keys]
    sort_cols = ["RoundNumber", "Date"] if "Date" in df else ["RoundNumber"]
    return df.sort_values(sort_cols).drop_duplicates(keys, keep="last")[[*keys, *meta]]


def compute_driver_stats(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Classificação por piloto; trocas de equipe não duplicam o competidor."""
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()
    key = "DriverId" if "DriverId" in df else "FullName"
    stats = df.groupby(key, as_index=False).agg(
        Races=("Started", "sum"),
        Points=("Points", "sum"),
        Wins=("OfficialFinish", lambda s: int((s == 1).sum())),
        Podiums=("OfficialFinish", lambda s: int((s <= 3).sum())),
        Poles=("OfficialGrid", lambda s: int((s == 1).sum())),
        DNFs=("DNF", "sum"),
        AvgFinish=("OfficialFinish", "mean"),
        AvgGrid=("OfficialGrid", "mean"),
        BestFinish=("OfficialFinish", "min"),
        AvgGain=("Gain", "mean"),
    )
    season_sessions = races[races["Year"] == year].copy()
    season_sessions["Points"] = pd.to_numeric(
        season_sessions["Points"], errors="coerce"
    )
    official_points = season_sessions.groupby(key)["Points"].sum()
    stats["Points"] = stats[key].map(official_points).fillna(stats["Points"])
    stats = stats.merge(_latest_metadata(df, [key]), on=key, how="left")
    stats["PodiumRate"] = stats["Podiums"].div(stats["Races"].replace(0, pd.NA))
    stats["DNFRate"] = stats["DNFs"].div(stats["Races"].replace(0, pd.NA))
    return _rank_by(stats)


def compute_team_stats(races: pd.DataFrame, year: int) -> pd.DataFrame:
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()
    key = "TeamId" if "TeamId" in df else "TeamName"
    stats = df.groupby(key, as_index=False).agg(
        Points=("Points", "sum"),
        Wins=("OfficialFinish", lambda s: int((s == 1).sum())),
        Podiums=("OfficialFinish", lambda s: int((s <= 3).sum())),
        Starts=("Started", "sum"),
        DNFs=("DNF", "sum"),
    )
    season_sessions = races[races["Year"] == year].copy()
    season_sessions["Points"] = pd.to_numeric(
        season_sessions["Points"], errors="coerce"
    )
    official_points = season_sessions.groupby(key)["Points"].sum()
    stats["Points"] = stats[key].map(official_points).fillna(stats["Points"])
    stats = stats.merge(
        _latest_metadata(df, [key], ("TeamName", "TeamColor")),
        on=key,
        how="left",
    )
    stats["Reliability"] = 1 - stats["DNFs"].div(stats["Starts"].replace(0, pd.NA))
    stats = stats.sort_values(
        ["Points", "TeamName"], ascending=[False, True]
    ).reset_index(drop=True)
    stats.insert(0, "Rank", stats.index + 1)
    return stats


def compute_reliability(races: pd.DataFrame, year: int) -> pd.DataFrame:
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()
    key = "DriverId" if "DriverId" in df else "FullName"
    out = df.groupby(key, as_index=False).agg(
        Starts=("Started", "sum"),
        DNFs=("DNF", "sum"),
        Points=("Points", "sum"),
        PointsFinishes=("Points", lambda s: int((s > 0).sum())),
        AvgGrid=("OfficialGrid", "mean"),
        BestGrid=("OfficialGrid", "min"),
        AvgGain=("Gain", "mean"),
        GainStd=("Gain", "std"),
    )
    out = out.merge(_latest_metadata(df, [key]), on=key, how="left")
    out["DNFRate"] = out["DNFs"].div(out["Starts"].replace(0, pd.NA))
    out["PointsFinishRate"] = out["PointsFinishes"].div(out["Starts"].replace(0, pd.NA))
    return out.sort_values("Points", ascending=False).reset_index(drop=True)


def compute_standings_history(races: pd.DataFrame, year: int) -> pd.DataFrame:
    """Snapshot por rodada, carregando pontos para ausências posteriores."""
    df = _season_races(races, year)
    if df.empty:
        return pd.DataFrame()
    key = "DriverId" if "DriverId" in df else "FullName"
    rounds = sorted(df["RoundNumber"].dropna().astype(int).unique())
    drivers = df[[key]].drop_duplicates().assign(_join=1)
    grid = drivers.merge(
        pd.DataFrame({"RoundNumber": rounds, "_join": 1}), on="_join"
    ).drop(columns="_join")
    season_sessions = races[races["Year"] == year].copy()
    season_sessions["Points"] = pd.to_numeric(
        season_sessions["Points"], errors="coerce"
    )
    points = season_sessions.groupby([key, "RoundNumber"], as_index=False)[
        "Points"
    ].sum()
    out = grid.merge(points, on=[key, "RoundNumber"], how="left").fillna({"Points": 0})
    out = out.sort_values([key, "RoundNumber"])
    out["CumulativePoints"] = out.groupby(key)["Points"].cumsum()
    out["ChampionshipRank"] = out.groupby("RoundNumber")["CumulativePoints"].rank(
        method="min", ascending=False
    )
    meta = _latest_metadata(df, [key])
    event_columns = [column for column in ("EventName", "Date") if column in df.columns]
    events = df.sort_values("RoundNumber").drop_duplicates("RoundNumber")[
        ["RoundNumber", *event_columns]
    ]
    if "EventName" not in events:
        events["EventName"] = events["RoundNumber"].map(lambda value: f"R{int(value)}")
    if "Date" not in events:
        events["Date"] = pd.NaT
    return out.merge(meta, on=key, how="left").merge(
        events, on="RoundNumber", how="left"
    )


def result_matrix(races: pd.DataFrame, year: int) -> pd.DataFrame:
    df = _season_races(races, year)
    if df.empty:
        return df
    df["ResultLabel"] = df["OfficialFinish"].map(
        lambda x: f"P{int(x)}" if pd.notna(x) else ""
    )
    df.loc[df["ResultStatus"] != "FINISHED", "ResultLabel"] = df["ResultStatus"]
    return df.sort_values(["RoundNumber", "Position"])


def summarize_race(results: pd.DataFrame, round_number: int) -> dict:
    """Resume uma corrida já normalizada para uso em cards editoriais."""
    if results.empty or "RoundNumber" not in results:
        return {}
    race = results[results["RoundNumber"] == round_number].copy()
    if race.empty:
        return {}

    race["OfficialFinish"] = pd.to_numeric(race.get("OfficialFinish"), errors="coerce")
    race["OfficialGrid"] = pd.to_numeric(race.get("OfficialGrid"), errors="coerce")
    classified = race[race["OfficialFinish"].notna()].sort_values("OfficialFinish")
    podium = classified[classified["OfficialFinish"] <= 3]
    gains = classified.dropna(subset=["OfficialGrid"]).copy()
    gains["Gain"] = gains["OfficialGrid"] - gains["OfficialFinish"]
    gains = gains[gains["Gain"] > 0]
    biggest_gainer = gains.sort_values(
        ["Gain", "OfficialFinish"], ascending=[False, True]
    ).head(1)

    event_name = race.get("EventName", pd.Series(dtype="object")).dropna()
    winner = classified[classified["OfficialFinish"] == 1].head(1)
    status = race.get("ResultStatus", pd.Series("FINISHED", index=race.index))
    incidents = int((status != "FINISHED").sum())
    return {
        "event_name": event_name.iloc[0]
        if not event_name.empty
        else f"Rodada {round_number}",
        "winner": winner.iloc[0]["FullName"] if not winner.empty else "—",
        "podium": " · ".join(
            f"P{int(row['OfficialFinish'])} {short_name(row['FullName'])}"
            for _, row in podium.iterrows()
        )
        or "—",
        "biggest_gainer": (
            f"{short_name(biggest_gainer.iloc[0]['FullName'])} "
            f"({biggest_gainer.iloc[0]['Gain']:+.0f})"
            if not biggest_gainer.empty
            else "—"
        ),
        "incidents": incidents,
        "classified": len(classified),
    }


def compute_teammate_h2h(races: pd.DataFrame, year: int) -> pd.DataFrame:
    df = _season_races(races, year)
    if df.empty or "TeamId" not in df.columns:
        return pd.DataFrame()
    rows = []
    season_sessions = races[races["Year"] == year].copy()
    season_sessions["Points"] = pd.to_numeric(
        season_sessions["Points"], errors="coerce"
    )
    for team_id, team_df in df.groupby("TeamId"):
        points = (
            season_sessions[season_sessions["TeamId"] == team_id]
            .groupby("DriverId")["Points"]
            .sum()
        )
        drivers = sorted(points.index, key=lambda d: (-points[d], d))
        if len(drivers) != 2:
            continue
        a, b = drivers
        latest = team_df.sort_values("RoundNumber").iloc[-1]
        abbr = team_df.drop_duplicates("DriverId", keep="last").set_index("DriverId")[
            "Abbreviation"
        ]
        rec = {
            "TeamId": team_id,
            "TeamName": latest["TeamName"],
            "TeamColor": latest["TeamColor"],
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
            pos = rnd.set_index("DriverId")["OfficialFinish"]
            grid = rnd.set_index("DriverId")["OfficialGrid"]
            if a not in pos.index or b not in pos.index:
                continue
            rec["Rounds"] += 1
            pa = pos[a] if pd.notna(pos[a]) else math.inf
            pb = pos[b] if pd.notna(pos[b]) else math.inf
            rec["RaceWinsA"] += int(pa < pb)
            rec["RaceWinsB"] += int(pb < pa)
            if pd.notna(grid[a]) and pd.notna(grid[b]):
                rec["QualiWinsA"] += int(grid[a] < grid[b])
                rec["QualiWinsB"] += int(grid[b] < grid[a])
        rows.append(rec)
    return (
        pd.DataFrame(rows)
        .sort_values("PointsA", ascending=False)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def normalize_probabilities(
    preds: pd.DataFrame, score_col: str = "raw_score"
) -> pd.DataFrame:
    """Normaliza escores por snapshot; um campeonato mutuamente exclusivo soma 1."""
    out = preds.copy()
    if out.empty:
        return out
    source = score_col if score_col in out else "prob_win"
    score = pd.to_numeric(out[source], errors="coerce").clip(lower=0)
    totals = score.groupby(out["dt_ref"]).transform("sum")
    sizes = out["dt_ref"].groupby(out["dt_ref"]).transform("size")
    out["raw_score"] = score
    normalized = score.div(totals.where(totals > 0))
    out["prob_win"] = normalized.where(totals > 0, 1 / sizes)
    return out


def _slope(y) -> float:
    ys = [float(v) for v in y if pd.notna(v)]
    if len(ys) < 2:
        return 0.0
    xs = range(len(ys))
    mx, my = sum(xs) / len(ys), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (v - my) for x, v in zip(xs, ys)) / den


def compute_momentum(
    preds: pd.DataFrame, last_n: int = 5, flat_eps: float = 0.005
) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    dates = sorted(preds["dt_ref"].unique())
    latest_date, prev_date = dates[-1], dates[-2] if len(dates) >= 2 else None
    rows = []
    meta_cols = [
        c
        for c in ("FullName", "Abbreviation", "TeamName", "TeamColor", "HeadshotUrl")
        if c in preds
    ]
    for driver_id, group in preds.sort_values("dt_ref").groupby("DriverId"):
        series = group.set_index("dt_ref")["prob_win"]
        latest = series.get(latest_date, float("nan"))
        prev = (
            series.get(prev_date, float("nan"))
            if prev_date is not None
            else float("nan")
        )
        slope = _slope(series.tail(last_n))
        rows.append(
            {
                "DriverId": driver_id,
                **{c: group[c].iloc[-1] for c in meta_cols},
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
    """Fallback descritivo; não é apresentado como explicação causal."""
    rows = []
    for feat, importance in sorted(
        importances.items(), key=lambda item: item[1], reverse=True
    ):
        if feat not in feature_row.index:
            continue
        value, median = feature_row[feat], field_median.get(feat, float("nan"))
        direction = (
            "—"
            if pd.isna(value) or pd.isna(median)
            else "acima"
            if value > median
            else "abaixo"
            if value < median
            else "na mediana"
        )
        rows.append(
            {
                "feature": feat,
                "importance": importance,
                "value": value,
                "field_median": median,
                "vs_field": direction,
            }
        )
        if len(rows) == k:
            break
    return pd.DataFrame(rows)


def build_insights(
    momentum: pd.DataFrame,
    h2h: pd.DataFrame,
    reliability: pd.DataFrame,
    last_n: int = 5,
) -> list[str]:
    out: list[str] = []
    if not momentum.empty and momentum["delta_prev"].notna().any():
        mover = momentum.loc[momentum["delta_prev"].abs().idxmax()]
        out.append(
            f"**{mover.get('FullName', mover['DriverId'])}** teve a maior mudança desde a rodada anterior ({mover['delta_prev'] * 100:+.1f} pp)."
        )
    if not momentum.empty and (momentum["slope"] > 0).any():
        rising = momentum.sort_values("slope", ascending=False).iloc[0]
        out.append(
            f"**{rising.get('FullName', rising['DriverId'])}** tem a tendência mais positiva nas últimas {last_n} rodadas ({rising['slope'] * 100:+.1f} pp/rodada)."
        )
    if not h2h.empty:
        close = (
            h2h.assign(margin=(h2h["RaceWinsA"] - h2h["RaceWinsB"]).abs())
            .query("Rounds >= 3")
            .sort_values(["margin", "Rounds"], ascending=[True, False])
        )
        if not close.empty:
            row = close.iloc[0]
            out.append(
                f"Duelo mais equilibrado: **{row['abbr_a']} {int(row['RaceWinsA'])}–{int(row['RaceWinsB'])} {row['abbr_b']}**, na {row['TeamName']}."
            )
    if not reliability.empty:
        valid = reliability.query("Starts >= 3")
        if not valid.empty:
            qualifier = valid.sort_values("AvgGrid").iloc[0]
            out.append(
                f"Melhor média de largada: **{qualifier['FullName']}**, P{qualifier['AvgGrid']:.1f}."
            )
    return out[:4]
