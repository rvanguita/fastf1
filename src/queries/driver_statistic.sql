WITH dim_dates AS (
    SELECT 
        DISTINCT date(Date) AS dt_ref
        , Year AS ref_year
    FROM results
    WHERE Year BETWEEN (SELECT MIN(Year) FROM results) AND '{year_stop}'
    -- WHERE Year BETWEEN ('{year_start}') AND ('{year_stop}')
),
interval_sessions AS (
    SELECT 
        d.ref_year
        , d.dt_ref
        , r.*
    FROM dim_dates d
    INNER JOIN results AS r
        ON date(r.Date) <= d.dt_ref
),
distinct_rounds AS (
    SELECT 
        DISTINCT dt_ref
        , Year
        , RoundNumber
    FROM interval_sessions
),
ranked_rounds AS (
    SELECT 
        Year
        , dt_ref
        , roundnumber
        , ROW_NUMBER() OVER (PARTITION BY dt_ref ORDER BY Year DESC, roundnumber DESC) AS rn
    FROM distinct_rounds
),
eligible_drivers AS (
    SELECT 
        DISTINCT i.dt_ref
        , i.driverid
    FROM interval_sessions i
    -- Candidate set: only drivers who have already participated in the
    -- reference season. Historical results still feed their rolling features.
    WHERE Year = ref_year
    ORDER BY dt_ref DESC
),
last_rounds AS (
    SELECT 
        Year
        , dt_ref
        , roundnumber
    FROM ranked_rounds
    WHERE rn <= '{last_rounds}'
    ORDER BY Year DESC
),
tb_results AS (
    SELECT i.*
    FROM interval_sessions i
    INNER JOIN eligible_drivers e
        ON i.dt_ref = e.dt_ref 
        AND i.driverid = e.driverid
    INNER JOIN last_rounds l
        ON i.dt_ref = l.dt_ref 
        AND i.year = l.year 
        AND i.roundnumber = l.roundnumber
),
tb_statistic AS (
SELECT 
    dt_ref
    , DriverId
    , COUNT(DISTINCT Year) AS qty_seasons
    , COUNT(*) AS qty_sessions
    , SUM(CASE WHEN (Status = 'Finished' OR Status Like '+%') THEN 1 ELSE 0 END) AS qty_sessions_finished
    , SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) AS qty_1place
    , SUM(CASE WHEN Position = 2 THEN 1 ELSE 0 END) AS qty_2place
    , SUM(CASE WHEN Position = 3 THEN 1 ELSE 0 END) AS qty_3place
    , SUM(CASE WHEN Position <= 3 THEN 1 ELSE 0 END) AS qty_place
    , SUM(Points) as total_points
    , ROUND(AVG(CASE WHEN GridPosition > 0 THEN GridPosition END), 2) AS avg_gridposition
    , ROUND(AVG(CASE WHEN ClassifiedPosition RLIKE '^[0-9]+$' THEN Position END), 2) AS avg_position
    , SUM(CASE WHEN GridPosition = 1 THEN 1 ELSE 0 END) AS qty_gridposition_1
    , SUM(CASE WHEN GridPosition = 1 AND Position = 1 THEN 1 ELSE 0 END) AS qty_poli_win
    , SUM(CASE WHEN Points > 0 THEN 1 ELSE 0 END) AS qty_sessions_with_points
    , SUM(CASE WHEN Position < GridPosition THEN 1 ELSE 0 END) AS qty_sessions_with_overtake
    , ROUND(AVG(CASE WHEN GridPosition > 0 AND ClassifiedPosition RLIKE '^[0-9]+$' THEN GridPosition - Position END), 2) AS avg_overtake
    , SUM(CASE WHEN Position <= 5 THEN 1 ELSE 0 END) AS qty_pos5
    , SUM(CASE WHEN GridPosition <= 5 THEN 1 ELSE 0 END) AS qty_gridpos5

    , SUM(CASE WHEN Mode = 'Race' THEN 1 ELSE 0 END) AS qty_sessions_r
    , SUM(CASE WHEN Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_sessions_s
    , SUM(CASE WHEN (Status = 'Finished' OR Status Like '+%') AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_sessions_finished_r
    , SUM(CASE WHEN (Status = 'Finished' OR Status Like '+%') AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_sessions_finished_s
    , SUM(CASE WHEN Position = 1 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_1place_r
    , SUM(CASE WHEN Position = 1 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_1place_s
    , SUM(CASE WHEN Position <= 3 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_place_r
    , SUM(CASE WHEN Position <= 3 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_place_s
    , SUM(CASE WHEN Mode = 'Race' THEN Points ELSE 0 END) as total_points_r
    , SUM(CASE WHEN Mode = 'Sprint' THEN Points ELSE 0 END) as total_points_s
    , ROUND(AVG(CASE WHEN Mode = 'Race' AND GridPosition > 0 THEN GridPosition END), 2) AS avg_gridposition_r
    , ROUND(AVG(CASE WHEN Mode = 'Sprint' AND GridPosition > 0 THEN GridPosition END), 2) AS avg_gridposition_s
    , ROUND(AVG(CASE WHEN Mode = 'Race' AND ClassifiedPosition RLIKE '^[0-9]+$' THEN Position END), 2) AS avg_position_r
    , ROUND(AVG(CASE WHEN Mode = 'Sprint' AND ClassifiedPosition RLIKE '^[0-9]+$' THEN Position END), 2) AS avg_position_s
    , SUM(CASE WHEN GridPosition = 1 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_gridposition_1_r
    , SUM(CASE WHEN GridPosition = 1 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_gridposition_1_s
    , SUM(CASE WHEN GridPosition = 1 AND Position = 1 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_poli_win_r
    , SUM(CASE WHEN GridPosition = 1 AND Position = 1 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_poli_win_s
    , SUM(CASE WHEN Points > 0 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_sessions_with_points_r
    , SUM(CASE WHEN Points > 0 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_sessions_with_points_s
    , SUM(CASE WHEN Position < GridPosition AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_sessions_with_overtake_r
    , SUM(CASE WHEN Position < GridPosition AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_sessions_with_overtake_s
    , ROUND(AVG(CASE WHEN Mode = 'Race' AND GridPosition > 0 AND ClassifiedPosition RLIKE '^[0-9]+$' THEN GridPosition - Position END), 2) AS avg_overtake_r
    , ROUND(AVG(CASE WHEN Mode = 'Sprint' AND GridPosition > 0 AND ClassifiedPosition RLIKE '^[0-9]+$' THEN GridPosition - Position END), 2) AS avg_overtake_s
    , SUM(CASE WHEN Position <= 5 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_pos5_r
    , SUM(CASE WHEN Position <= 5 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_pos5_S
    , SUM(CASE WHEN GridPosition <= 5 AND Mode = 'Race' THEN 1 ELSE 0 END) AS qty_gridpos5_r
    , SUM(CASE WHEN GridPosition <= 5 AND Mode = 'Sprint' THEN 1 ELSE 0 END) AS qty_gridpos5_S
FROM tb_results
GROUP BY dt_ref, DriverId
)

SELECT *
FROM tb_statistic
ORDER BY dt_ref desc, DriverId
