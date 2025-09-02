-- Advanced SQL Queries for Spotify Analytics
-- Complex analysis using CTEs, Window Functions, and Advanced Techniques

-- =============================================================================
-- 1. ADVANCED SESSIONIZATION WITH WINDOW FUNCTIONS
-- =============================================================================

-- Create sessions based on 30-minute gaps using window functions
WITH session_breaks AS (
    SELECT *,
           LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
           CASE 
               WHEN timestamp - LAG(timestamp) OVER (ORDER BY timestamp) > INTERVAL '30 minutes' 
                    OR LAG(timestamp) OVER (ORDER BY timestamp) IS NULL
               THEN 1 
               ELSE 0 
           END as is_session_start
    FROM spotify_cleaned
),
session_ids AS (
    SELECT *,
           SUM(is_session_start) OVER (ORDER BY timestamp) as session_id
    FROM session_breaks
)
SELECT 
    session_id,
    MIN(timestamp) as session_start,
    MAX(timestamp) as session_end,
    COUNT(*) as tracks_in_session,
    ROUND(SUM(seconds_played)/60, 1) as total_session_minutes,
    ROUND(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/60, 1) as session_duration_minutes,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as session_skip_rate,
    MODE() WITHIN GROUP (ORDER BY platform) as primary_platform
FROM session_ids
GROUP BY session_id
HAVING COUNT(*) >= 2  -- Sessions with at least 2 tracks
ORDER BY session_id
LIMIT 50;

-- =============================================================================
-- 2. COHORT ANALYSIS FOR USER RETENTION
-- =============================================================================

-- User cohort analysis by first listening date
WITH user_first_date AS (
    SELECT 
        spotify_track_uri as user_proxy,
        DATE_TRUNC('week', MIN(timestamp)) as cohort_week
    FROM spotify_cleaned
    GROUP BY spotify_track_uri
),
user_activity AS (
    SELECT 
        s.spotify_track_uri,
        u.cohort_week,
        DATE_TRUNC('week', s.timestamp) as activity_week,
        EXTRACT(WEEK FROM s.timestamp) - EXTRACT(WEEK FROM u.cohort_week) as weeks_since_first
    FROM spotify_cleaned s
    JOIN user_first_date u ON s.spotify_track_uri = u.user_proxy
)
SELECT 
    cohort_week,
    weeks_since_first,
    COUNT(DISTINCT spotify_track_uri) as active_users,
    ROUND(COUNT(DISTINCT spotify_track_uri) * 100.0 / 
          FIRST_VALUE(COUNT(DISTINCT spotify_track_uri)) 
          OVER (PARTITION BY cohort_week ORDER BY weeks_since_first), 1) as retention_rate
FROM user_activity
WHERE weeks_since_first <= 8  -- First 8 weeks
GROUP BY cohort_week, weeks_since_first
ORDER BY cohort_week, weeks_since_first;

-- =============================================================================
-- 3. ROLLING METRICS AND TREND ANALYSIS
-- =============================================================================

-- 7-day rolling averages for key metrics
WITH daily_metrics AS (
    SELECT 
        date,
        COUNT(*) as daily_plays,
        ROUND(SUM(seconds_played)/3600, 1) as daily_hours,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as daily_skip_rate,
        COUNT(DISTINCT spotify_track_uri) as unique_tracks
    FROM spotify_cleaned
    GROUP BY date
)
SELECT 
    date,
    daily_plays,
    ROUND(AVG(daily_plays) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) as rolling_7d_plays,
    daily_hours,
    ROUND(AVG(daily_hours) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) as rolling_7d_hours,
    daily_skip_rate,
    ROUND(AVG(daily_skip_rate) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) as rolling_7d_skip_rate,
    -- Week-over-week growth
    ROUND(((daily_plays - LAG(daily_plays, 7) OVER (ORDER BY date)) * 100.0 / 
           NULLIF(LAG(daily_plays, 7) OVER (ORDER BY date), 0)), 1) as wow_plays_growth
FROM daily_metrics
ORDER BY date;

-- =============================================================================
-- 4. ADVANCED RANKING AND PERCENTILE ANALYSIS
-- =============================================================================

-- Artist performance with rankings and percentiles
WITH artist_metrics AS (
    SELECT 
        artist_name,
        COUNT(*) as total_plays,
        ROUND(SUM(seconds_played)/3600, 1) as total_hours,
        ROUND(AVG(seconds_played), 1) as avg_seconds_per_play,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct,
        ROUND(AVG(percent_played), 1) as avg_percent_played,
        COUNT(DISTINCT track_name) as unique_tracks
    FROM spotify_cleaned
    GROUP BY artist_name
    HAVING COUNT(*) >= 10
)
SELECT 
    artist_name,
    total_plays,
    RANK() OVER (ORDER BY total_plays DESC) as plays_rank,
    NTILE(5) OVER (ORDER BY total_plays) as plays_quintile,
    total_hours,
    RANK() OVER (ORDER BY total_hours DESC) as hours_rank,
    skip_rate_pct,
    RANK() OVER (ORDER BY skip_rate_pct ASC) as engagement_rank,  -- Lower skip rate = better engagement
    ROUND(PERCENT_RANK() OVER (ORDER BY skip_rate_pct DESC) * 100, 1) as skip_rate_percentile,
    unique_tracks,
    -- Composite engagement score
    ROUND((PERCENT_RANK() OVER (ORDER BY total_plays) * 0.3 +
           PERCENT_RANK() OVER (ORDER BY avg_percent_played) * 0.4 +
           PERCENT_RANK() OVER (ORDER BY skip_rate_pct DESC) * 0.3) * 100, 1) as engagement_score
FROM artist_metrics
ORDER BY engagement_score DESC
LIMIT 20;

-- =============================================================================
-- 5. COMPLEX AGGREGATIONS WITH CUBE AND ROLLUP
-- =============================================================================

-- Multi-dimensional analysis with ROLLUP
SELECT 
    COALESCE(platform, 'ALL PLATFORMS') as platform,
    COALESCE(time_of_day, 'ALL TIMES') as time_of_day,
    COUNT(*) as total_plays,
    ROUND(SUM(seconds_played)/3600, 1) as total_hours,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct,
    ROUND(AVG(percent_played), 1) as avg_percent_played
FROM spotify_cleaned
GROUP BY ROLLUP(platform, time_of_day)
ORDER BY platform NULLS LAST, time_of_day NULLS LAST;

-- =============================================================================
-- 6. PREDICTIVE ANALYTICS FEATURES
-- =============================================================================

-- Create features for skip prediction model
WITH track_history AS (
    SELECT 
        spotify_track_uri,
        track_name,
        artist_name,
        COUNT(*) as track_play_count,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as historical_skip_rate,
        ROUND(AVG(seconds_played), 1) as avg_play_duration,
        MAX(estimated_track_length_ms)/1000 as track_length_seconds
    FROM spotify_cleaned
    GROUP BY spotify_track_uri, track_name, artist_name
),
user_behavior AS (
    SELECT 
        spotify_track_uri as user_proxy,
        COUNT(*) as user_total_plays,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as user_skip_tendency,
        ROUND(AVG(percent_played), 1) as user_avg_completion,
        MODE() WITHIN GROUP (ORDER BY platform) as preferred_platform,
        ROUND(AVG(hour_of_day), 1) as avg_listening_hour
    FROM spotify_cleaned
    GROUP BY spotify_track_uri
)
SELECT 
    s.*,
    th.track_play_count,
    th.historical_skip_rate,
    th.track_length_seconds,
    ub.user_total_plays,
    ub.user_skip_tendency,
    ub.user_avg_completion,
    ub.preferred_platform,
    -- Risk factors
    CASE 
        WHEN th.historical_skip_rate > 70 THEN 'High Risk'
        WHEN th.historical_skip_rate > 40 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as track_skip_risk,
    CASE 
        WHEN ub.user_skip_tendency > 60 THEN 'High Skip User'
        WHEN ub.user_skip_tendency > 30 THEN 'Medium Skip User'
        ELSE 'Low Skip User'
    END as user_skip_profile
FROM spotify_cleaned s
JOIN track_history th ON s.spotify_track_uri = th.spotify_track_uri 
                      AND s.track_name = th.track_name 
                      AND s.artist_name = th.artist_name
JOIN user_behavior ub ON s.spotify_track_uri = ub.user_proxy
ORDER BY s.timestamp DESC
LIMIT 1000;

-- =============================================================================
-- 7. BUSINESS INTELLIGENCE DASHBOARD QUERIES
-- =============================================================================

-- Executive KPI Dashboard
WITH kpi_metrics AS (
    SELECT 
        COUNT(*) as total_streams,
        COUNT(DISTINCT spotify_track_uri) as unique_tracks,
        COUNT(DISTINCT artist_name) as unique_artists,
        ROUND(SUM(seconds_played)/3600, 0) as total_hours,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as overall_skip_rate,
        ROUND(AVG(percent_played), 1) as avg_completion_rate,
        COUNT(DISTINCT date) as active_days
    FROM spotify_cleaned
)
SELECT 
    'Total Streams' as metric, CAST(total_streams as TEXT) as value
FROM kpi_metrics
UNION ALL
SELECT 
    'Content Catalog', CAST(unique_tracks as TEXT) || ' tracks, ' || CAST(unique_artists as TEXT) || ' artists'
FROM kpi_metrics
UNION ALL
SELECT 
    'Total Listening Hours', CAST(total_hours as TEXT) || ' hours'
FROM kpi_metrics
UNION ALL
SELECT 
    'Skip Rate', CAST(overall_skip_rate as TEXT) || '%'
FROM kpi_metrics
UNION ALL
SELECT 
    'Completion Rate', CAST(avg_completion_rate as TEXT) || '%'
FROM kpi_metrics
UNION ALL
SELECT 
    'Daily Avg Streams', CAST(ROUND(total_streams * 1.0 / active_days, 0) as TEXT)
FROM kpi_metrics;
