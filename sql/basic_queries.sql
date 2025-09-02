-- Spotify Analytics SQL Queries
-- Basic Analysis Queries for Product Analytics

-- =============================================================================
-- 1. TOP CONTENT ANALYSIS
-- =============================================================================

-- Top 20 tracks by total play count
SELECT 
    track_name,
    artist_name,
    COUNT(*) as play_count,
    ROUND(SUM(seconds_played)/60, 1) as total_minutes,
    ROUND(AVG(percent_played), 1) as avg_percent_played,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY track_name, artist_name
HAVING play_count >= 5
ORDER BY play_count DESC
LIMIT 20;

-- Top 15 artists by total listening time
SELECT 
    artist_name,
    COUNT(*) as total_plays,
    COUNT(DISTINCT track_name) as unique_tracks,
    ROUND(SUM(seconds_played)/3600, 1) as total_hours,
    ROUND(AVG(seconds_played), 1) as avg_seconds_per_play,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY artist_name
HAVING total_plays >= 10
ORDER BY total_hours DESC
LIMIT 15;

-- =============================================================================
-- 2. PLATFORM ANALYSIS
-- =============================================================================

-- Platform performance comparison
SELECT 
    platform,
    COUNT(*) as total_plays,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as market_share_pct,
    ROUND(SUM(seconds_played)/3600, 1) as total_hours,
    ROUND(AVG(seconds_played), 1) as avg_seconds_per_play,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct,
    ROUND(AVG(percent_played), 1) as avg_percent_played
FROM spotify_cleaned
GROUP BY platform
ORDER BY total_plays DESC;

-- Platform usage by time of day
SELECT 
    time_of_day,
    platform,
    COUNT(*) as plays,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY time_of_day), 1) as platform_share_pct
FROM spotify_cleaned
GROUP BY time_of_day, platform
ORDER BY time_of_day, plays DESC;

-- =============================================================================
-- 3. TEMPORAL ANALYSIS
-- =============================================================================

-- Peak usage analysis by hour
SELECT 
    hour_of_day,
    COUNT(*) as total_plays,
    ROUND(SUM(seconds_played)/3600, 1) as total_hours,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct,
    ROUND(AVG(percent_played), 1) as avg_percent_played
FROM spotify_cleaned
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Day of week patterns
SELECT 
    day_of_week,
    COUNT(*) as total_plays,
    ROUND(SUM(seconds_played)/3600, 1) as total_hours,
    ROUND(AVG(seconds_played), 1) as avg_seconds_per_play,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY day_of_week, day_of_week_num
ORDER BY day_of_week_num;

-- =============================================================================
-- 4. SKIP BEHAVIOR ANALYSIS
-- =============================================================================

-- Skip rate analysis by various dimensions
SELECT 
    'Overall' as dimension,
    'All' as category,
    COUNT(*) as total_plays,
    SUM(CASE WHEN is_skip = 'TRUE' THEN 1 ELSE 0 END) as skips,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned

UNION ALL

SELECT 
    'Platform' as dimension,
    platform as category,
    COUNT(*) as total_plays,
    SUM(CASE WHEN is_skip = 'TRUE' THEN 1 ELSE 0 END) as skips,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY platform

UNION ALL

SELECT 
    'Time of Day' as dimension,
    time_of_day as category,
    COUNT(*) as total_plays,
    SUM(CASE WHEN is_skip = 'TRUE' THEN 1 ELSE 0 END) as skips,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY time_of_day

UNION ALL

SELECT 
    'Shuffle Mode' as dimension,
    CASE WHEN shuffle = 'TRUE' THEN 'Shuffle On' ELSE 'Shuffle Off' END as category,
    COUNT(*) as total_plays,
    SUM(CASE WHEN is_skip = 'TRUE' THEN 1 ELSE 0 END) as skips,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct
FROM spotify_cleaned
GROUP BY shuffle

ORDER BY dimension, skip_rate_pct DESC;

-- Most skipped tracks (minimum 10 plays)
SELECT 
    track_name,
    artist_name,
    COUNT(*) as total_plays,
    SUM(CASE WHEN is_skip = 'TRUE' THEN 1 ELSE 0 END) as total_skips,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as skip_rate_pct,
    ROUND(AVG(percent_played), 1) as avg_percent_played
FROM spotify_cleaned
GROUP BY track_name, artist_name
HAVING total_plays >= 10
ORDER BY skip_rate_pct DESC, total_plays DESC
LIMIT 15;

-- =============================================================================
-- 5. USER ENGAGEMENT METRICS
-- =============================================================================

-- Daily activity summary
SELECT 
    date,
    COUNT(*) as daily_plays,
    COUNT(DISTINCT spotify_track_uri) as unique_tracks,
    ROUND(SUM(seconds_played)/3600, 1) as daily_hours,
    ROUND(AVG(seconds_played), 1) as avg_seconds_per_play,
    ROUND(AVG(CASE WHEN is_skip = 'TRUE' THEN 1.0 ELSE 0.0 END) * 100, 1) as daily_skip_rate
FROM spotify_cleaned
GROUP BY date
ORDER BY date;

-- Listening quality distribution
SELECT 
    listening_quality,
    COUNT(*) as plays,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage,
    ROUND(AVG(seconds_played), 1) as avg_seconds,
    ROUND(AVG(percent_played), 1) as avg_percent_played
FROM spotify_cleaned
GROUP BY listening_quality
ORDER BY 
    CASE listening_quality 
        WHEN 'High' THEN 1 
        WHEN 'Medium' THEN 2 
        WHEN 'Low' THEN 3 
    END;
