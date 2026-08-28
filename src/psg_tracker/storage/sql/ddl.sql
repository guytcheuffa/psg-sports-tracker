-- Schema DuckDB : tables typees pour matches et tirs (shots)
--
-- Deux sources coexistent (StatsBomb = corpus d'entrainement historique,
-- Understat = saison en cours scrapee) : la cle primaire est composite
-- (source, match_id) car chaque source a son propre espace d'identifiants.

CREATE TABLE IF NOT EXISTS matches (
    match_id     BIGINT,
    source       VARCHAR NOT NULL,  -- 'statsbomb' | 'understat'
    match_date   DATE,
    home_team    VARCHAR,
    away_team    VARCHAR,
    competition  VARCHAR,
    season       VARCHAR,
    PRIMARY KEY (source, match_id)
);

CREATE TABLE IF NOT EXISTS shots (
    event_id     VARCHAR PRIMARY KEY,  -- prefixe par source, deja unique globalement
    match_id     BIGINT,
    source       VARCHAR NOT NULL,     -- 'statsbomb' | 'understat'
    player_id    BIGINT,
    player_name  VARCHAR,
    team_id      BIGINT,
    minute       INTEGER,
    second       INTEGER,
    loc_x        DOUBLE,               -- referentiel StatsBomb (120x80), normalise a l'ingestion
    loc_y        DOUBLE,
    body_part    VARCHAR,
    shot_type    VARCHAR,
    outcome      VARCHAR,
    is_goal      BOOLEAN,
    FOREIGN KEY (source, match_id) REFERENCES matches(source, match_id)
);
