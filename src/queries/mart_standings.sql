WITH races AS (
    SELECT * FROM mart_driver_round
),
rounds AS (
    SELECT DISTINCT season, round_number, event_date, event_name FROM races
),
drivers AS (
    SELECT
        season,
        driver_id,
        max_by(driver_name, round_number) AS driver_name,
        max_by(abbreviation, round_number) AS abbreviation,
        max_by(team_id, round_number) AS team_id,
        max_by(team_name, round_number) AS team_name,
        max_by(team_color, round_number) AS team_color
    FROM races
    GROUP BY season, driver_id
),
driver_rounds AS (
    SELECT d.*, r.round_number, r.event_date, r.event_name
    FROM drivers d
    INNER JOIN rounds r ON d.season = r.season
),
round_points AS (
    SELECT
        Year AS season,
        CAST(RoundNumber AS INT) AS round_number,
        DriverId AS driver_id,
        SUM(CAST(Points AS DOUBLE)) AS points
    FROM results
    WHERE Mode IN ('Race', 'Sprint')
    GROUP BY Year, RoundNumber, DriverId
),
snapshots AS (
    SELECT
        d.*,
        COALESCE(p.points, 0) AS round_points,
        SUM(COALESCE(p.points, 0)) OVER (
            PARTITION BY d.season, d.driver_id ORDER BY d.round_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_points
    FROM driver_rounds d
    LEFT JOIN round_points p
      ON d.season = p.season
     AND d.round_number = p.round_number
     AND d.driver_id = p.driver_id
)
SELECT
    *,
    RANK() OVER (
        PARTITION BY season, round_number
        ORDER BY cumulative_points DESC
    ) AS championship_rank
FROM snapshots
