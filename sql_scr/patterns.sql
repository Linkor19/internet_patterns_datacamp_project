-- SELECT *
-- FROM patterns
-- LIMIT 10;


-------------------- most progressive countries for 5 last years
-- SELECT *
-- FROM (
--     SELECT "Country Name",
--         RANK() OVER(ORDER BY (CAST ("2022" - "2018" AS DECIMAL)) DESC) AS rnk,
--         CAST ("2022" - "2018" AS DECIMAL) AS "Progress in perc", "2018", "2022"
--     from patterns
--     WHERE "2018" IS NOT NULL AND "2022" IS NOT NULL) Q
-- WHERE rnk < 6;

--------------------the most advanced countries of the last 10 years

-- SELECT *
-- FROM (SELECT *,
--              RANK() OVER (ORDER BY Actual_Info DESC) AS rnk
--       FROM (SELECT "Country Name",
--                    CASE
--                        WHEN "2023" IS NOT NULL THEN "2023"
--                        WHEN "2022" IS NOT NULL THEN "2022"
--                        WHEN "2021" IS NOT NULL THEN "2021"
--                        WHEN "2020" IS NOT NULL THEN "2020"
--                        WHEN "2019" IS NOT NULL THEN "2019"
--                        ELSE NULL
--                        END AS Actual_Info
--             FROM patterns) Q
--       WHERE Actual_Info IS NOT NULL
-- )M
-- WHERE rnk < 11;

SELECT *
FROM edu
LIMIT 10