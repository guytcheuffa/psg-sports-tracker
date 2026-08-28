-- Transformations analytiques (vues) sur les tables brutes

CREATE OR REPLACE VIEW shots_enriched AS
SELECT
    s.*,
    m.match_date,
    m.competition,
    m.season,
    -- distance euclidienne au centre du but (120, 40) en referentiel StatsBomb
    SQRT(POWER(120 - s.loc_x, 2) + POWER(40 - s.loc_y, 2)) AS distance_to_goal
FROM shots s
JOIN matches m ON s.source = m.source AND s.match_id = m.match_id;
