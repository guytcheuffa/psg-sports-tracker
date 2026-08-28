-- Schema DuckDB : tables typees pour matches et tirs (shots)

CREATE TABLE IF NOT EXISTS matches (
    match_id     BIGINT PRIMARY KEY,
    match_date   DATE,
    home_team    VARCHAR,
    away_team    VARCHAR,
    competition  VARCHAR,
    season       VARCHAR
);

CREATE TABLE IF NOT EXISTS shots (
    event_id     VARCHAR PRIMARY KEY,
    match_id     BIGINT REFERENCES matches(match_id),
    player_id    BIGINT,
    player_name  VARCHAR,
    team_id      BIGINT,
    minute       INTEGER,
    second       INTEGER,
    loc_x        DOUBLE,
    loc_y        DOUBLE,
    body_part    VARCHAR,
    shot_type    VARCHAR,
    outcome      VARCHAR,
    is_goal      BOOLEAN
);
