WITH year_driver_points AS (
SELECT 
    Year
    , DriverId
    , SUM(Points) as total_points
    , SUM(CASE WHEN Mode = 'Race' AND Position = 1 THEN 1 ELSE 0 END) AS wins
    , SUM(CASE WHEN Mode = 'Race' AND Position = 2 THEN 1 ELSE 0 END) AS second_places
    , SUM(CASE WHEN Mode = 'Race' AND Position = 3 THEN 1 ELSE 0 END) AS third_places
FROM results
WHERE Year < YEAR(current_date())
GROUP BY 
    Year
    , DriverId
ORDER BY 
    Year
    , total_points DESC
),

rn_year_driver AS (
SELECT 
    *
    , ROW_NUMBER() OVER (
        PARTITION BY Year
        ORDER BY total_points DESC, wins DESC, second_places DESC, third_places DESC
    ) as rank_driver
FROM year_driver_points
)

SELECT
    *
FROM rn_year_driver
WHERE rank_driver = 1
