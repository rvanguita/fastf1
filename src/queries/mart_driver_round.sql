SELECT
    Year AS season,
    CAST(RoundNumber AS INT) AS round_number,
    CAST(Date AS DATE) AS event_date,
    EventName AS event_name,
    DriverId AS driver_id,
    FullName AS driver_name,
    Abbreviation AS abbreviation,
    TeamId AS team_id,
    TeamName AS team_name,
    TeamColor AS team_color,
    CAST(Points AS DOUBLE) AS points,
    CAST(Position AS INT) AS result_order,
    CASE WHEN GridPosition > 0 THEN CAST(GridPosition AS INT) END AS official_grid,
    CASE WHEN ClassifiedPosition RLIKE '^[0-9]+$' THEN CAST(ClassifiedPosition AS INT) END AS official_finish,
    CASE
        WHEN ClassifiedPosition RLIKE '^[0-9]+$' THEN 'FINISHED'
        WHEN ClassifiedPosition = 'R' THEN 'DNF'
        WHEN ClassifiedPosition IN ('D', 'E') THEN 'DSQ'
        WHEN ClassifiedPosition = 'W' THEN 'DNS'
        WHEN ClassifiedPosition = 'F' THEN 'DNQ'
        WHEN ClassifiedPosition = 'N' THEN 'NC'
        ELSE 'DNF'
    END AS result_status,
    CASE WHEN ClassifiedPosition NOT IN ('W', 'F') THEN TRUE ELSE FALSE END AS started
FROM results
WHERE Mode = 'Race'
