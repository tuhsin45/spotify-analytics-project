-- Sessionization SQL for Spotify Analytics
-- Advanced session analysis with gap-based sessionization

-- =============================================================================
-- 1. SESSION CREATION WITH 30-MINUTE GAP RULE
-- =============================================================================

-- Step 1: Identify session breaks
CREATE TEMPORARY VIEW session_breaks AS
SELECT 
    *,
    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
    timestamp - LAG(timestamp) OVER (ORDER BY timestamp) as time_since_last,
    CASE 
        WHEN timestamp - LAG(timestamp) OVER (ORDER BY timestamp) > INTERVAL '30 minutes' 
             OR LAG(timestamp) OVER (ORDER BY timestamp) IS NULL
        THEN 1 
        ELSE 0 
    END as is_session_start
FROM spotify_cleaned
ORDER BY timestamp;

-- Step 2: Assign session IDs
CREATE TEMPORARY VIEW sessions_with_ids AS
SELECT 
    *,
    SUM(is_session_start) OVER (ORDER BY timestamp ROWS UNBOUNDED PRECEDING) as session_id
FROM session_breaks;

-- =============================================================================
-- 2. SESSION-LEVEL METRICS
-- =============================================================================

-- Comprehensive session analysis
CREATE TEMPORARY VIEW session_metrics AS
SELECT 
    session_id,
    MIN(timestamp) as session_start,
    MAX(timestamp) as session_end,
    COUNT(*) as tracks_count,
    COUNT(DISTINCT spotify_track_uri) as unique_tracks,
    COUNT(DISTINCT artist_name) as unique_artists,
    
    -- Time metrics
    ROUND(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/60, 1) as session_duration_minutes,
    ROUND(SUM(seconds_played)/60, 1) as total_listening_minutes,
    ROUND(AVG(seconds_played), 1) as avg_track_duration,
    
    -- Engagement metrics
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as session_skip_rate,
    ROUND(AVG(percent_played), 1) as avg_percent_played,
    ROUND(SUM(seconds_played) / NULLIF(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))), 0) * 100, 1) as listening_efficiency,
    
    -- Platform and behavior
    MODE() WITHIN GROUP (ORDER BY platform) as primary_platform,
    ROUND(AVG(CASE WHEN shuffle = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as shuffle_usage_pct,
    
    -- Temporal features
    EXTRACT(HOUR FROM MIN(timestamp)) as session_start_hour,
    EXTRACT(DOW FROM MIN(timestamp)) as session_start_dow,
    CASE 
        WHEN EXTRACT(HOUR FROM MIN(timestamp)) BETWEEN 6 AND 11 THEN 'Morning'
        WHEN EXTRACT(HOUR FROM MIN(timestamp)) BETWEEN 12 AND 17 THEN 'Afternoon'
        WHEN EXTRACT(HOUR FROM MIN(timestamp)) BETWEEN 18 AND 21 THEN 'Evening'
        ELSE 'Night'
    END as session_time_period,
    
    -- Session quality indicators
    CASE 
        WHEN COUNT(*) >= 10 AND AVG(percent_played) >= 70 AND AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) <= 0.3 
        THEN 'High Quality'
        WHEN COUNT(*) >= 5 AND AVG(percent_played) >= 50 AND AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) <= 0.5 
        THEN 'Medium Quality'
        ELSE 'Low Quality'
    END as session_quality
    
FROM sessions_with_ids
GROUP BY session_id
HAVING COUNT(*) >= 2;  -- Only sessions with 2+ tracks

-- =============================================================================
-- 3. SESSION ANALYSIS QUERIES
-- =============================================================================

-- Session distribution analysis
SELECT 
    session_quality,
    COUNT(*) as session_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage,
    ROUND(AVG(tracks_count), 1) as avg_tracks,
    ROUND(AVG(session_duration_minutes), 1) as avg_duration_min,
    ROUND(AVG(session_skip_rate), 1) as avg_skip_rate,
    ROUND(AVG(total_listening_minutes), 1) as avg_listening_min
FROM session_metrics
GROUP BY session_quality
ORDER BY 
    CASE session_quality 
        WHEN 'High Quality' THEN 1 
        WHEN 'Medium Quality' THEN 2 
        WHEN 'Low Quality' THEN 3 
    END;

-- Platform session performance
SELECT 
    primary_platform,
    COUNT(*) as session_count,
    ROUND(AVG(tracks_count), 1) as avg_tracks_per_session,
    ROUND(AVG(session_duration_minutes), 1) as avg_session_duration,
    ROUND(AVG(session_skip_rate), 1) as avg_skip_rate,
    ROUND(AVG(total_listening_minutes), 1) as avg_listening_time,
    ROUND(AVG(listening_efficiency), 1) as avg_listening_efficiency,
    COUNT(CASE WHEN session_quality = 'High Quality' THEN 1 END) * 100.0 / COUNT(*) as high_quality_pct
FROM session_metrics
GROUP BY primary_platform
ORDER BY session_count DESC;

-- Session patterns by time of day
SELECT 
    session_time_period,
    session_start_hour,
    COUNT(*) as session_count,
    ROUND(AVG(tracks_count), 1) as avg_tracks,
    ROUND(AVG(session_duration_minutes), 1) as avg_duration,
    ROUND(AVG(session_skip_rate), 1) as avg_skip_rate,
    ROUND(COUNT(CASE WHEN session_quality = 'High Quality' THEN 1 END) * 100.0 / COUNT(*), 1) as high_quality_pct
FROM session_metrics
GROUP BY session_time_period, session_start_hour
ORDER BY session_start_hour;

-- =============================================================================
-- 4. SESSION JOURNEY ANALYSIS
-- =============================================================================

-- Track progression within sessions (first 10 tracks of each session)
WITH session_tracks AS (
    SELECT 
        session_id,
        timestamp,
        track_name,
        artist_name,
        is_skip,
        percent_played,
        reason_start,
        reason_end,
        ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) as track_position
    FROM sessions_with_ids
),
session_journey AS (
    SELECT 
        track_position,
        COUNT(*) as total_occurrences,
        ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_by_position,
        ROUND(AVG(percent_played), 1) as avg_completion_by_position,
        MODE() WITHIN GROUP (ORDER BY reason_start) as common_start_reason,
        MODE() WITHIN GROUP (ORDER BY reason_end) as common_end_reason
    FROM session_tracks
    WHERE track_position <= 10  -- First 10 tracks of each session
    GROUP BY track_position
)
SELECT 
    track_position,
    total_occurrences,
    skip_rate_by_position,
    avg_completion_by_position,
    common_start_reason,
    common_end_reason,
    -- Calculate engagement drop-off
    skip_rate_by_position - LAG(skip_rate_by_position) OVER (ORDER BY track_position) as skip_rate_change
FROM session_journey
ORDER BY track_position;

-- =============================================================================
-- 5. SESSION-BASED USER SEGMENTATION
-- =============================================================================

-- User behavior based on session patterns
WITH user_session_behavior AS (
    SELECT 
        primary_platform as user_platform,  -- Using platform as user proxy
        COUNT(*) as total_sessions,
        ROUND(AVG(tracks_count), 1) as avg_tracks_per_session,
        ROUND(AVG(session_duration_minutes), 1) as avg_session_duration,
        ROUND(AVG(session_skip_rate), 1) as avg_session_skip_rate,
        ROUND(AVG(total_listening_minutes), 1) as avg_listening_per_session,
        COUNT(CASE WHEN session_quality = 'High Quality' THEN 1 END) * 100.0 / COUNT(*) as high_quality_session_pct,
        ROUND(AVG(shuffle_usage_pct), 1) as avg_shuffle_usage,
        COUNT(DISTINCT session_time_period) as time_diversity,
        MODE() WITHIN GROUP (ORDER BY session_time_period) as preferred_time_period
    FROM session_metrics
    GROUP BY primary_platform
)
SELECT 
    user_platform,
    total_sessions,
    avg_tracks_per_session,
    avg_session_duration,
    avg_session_skip_rate,
    high_quality_session_pct,
    preferred_time_period,
    -- User segment classification
    CASE 
        WHEN total_sessions >= 50 AND high_quality_session_pct >= 60 THEN 'Power User'
        WHEN total_sessions >= 20 AND high_quality_session_pct >= 40 THEN 'Engaged User'
        WHEN total_sessions >= 10 AND avg_session_skip_rate <= 40 THEN 'Casual User'
        WHEN avg_session_skip_rate > 60 THEN 'At-Risk User'
        ELSE 'New/Light User'
    END as user_segment
FROM user_session_behavior
ORDER BY total_sessions DESC;

-- =============================================================================
-- 6. SESSION RETENTION ANALYSIS
-- =============================================================================

-- Daily session retention
WITH daily_sessions AS (
    SELECT 
        DATE(session_start) as session_date,
        primary_platform,
        COUNT(*) as daily_session_count,
        ROUND(AVG(session_duration_minutes), 1) as avg_daily_session_duration,
        ROUND(AVG(session_skip_rate), 1) as avg_daily_skip_rate
    FROM session_metrics
    GROUP BY DATE(session_start), primary_platform
),
session_retention AS (
    SELECT 
        session_date,
        daily_session_count,
        avg_daily_session_duration,
        avg_daily_skip_rate,
        LAG(daily_session_count) OVER (ORDER BY session_date) as prev_day_sessions,
        -- Calculate day-over-day change
        ROUND(((daily_session_count - LAG(daily_session_count) OVER (ORDER BY session_date)) * 100.0 / 
               NULLIF(LAG(daily_session_count) OVER (ORDER BY session_date), 0)), 1) as dod_session_change
    FROM daily_sessions
)
SELECT 
    session_date,
    daily_session_count,
    prev_day_sessions,
    dod_session_change,
    avg_daily_session_duration,
    avg_daily_skip_rate,
    -- 7-day rolling average
    ROUND(AVG(daily_session_count) OVER (ORDER BY session_date ROWS 6 PRECEDING), 1) as rolling_7d_sessions
FROM session_retention
ORDER BY session_date;

-- Clean up temporary views
DROP VIEW IF EXISTS session_breaks;
DROP VIEW IF EXISTS sessions_with_ids;
DROP VIEW IF EXISTS session_metrics;
