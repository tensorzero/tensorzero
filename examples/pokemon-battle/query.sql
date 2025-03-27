WITH unique_feedback AS (
    SELECT 
        ji.episode_id,
        bmf.tags['team_id'] as team_id,
        any(bmf.value) as value  -- Takes one value per episode_id
    FROM tensorzero.JsonInference ji
    LEFT JOIN tensorzero.BooleanMetricFeedback bmf 
        ON ji.episode_id = bmf.target_id
    WHERE bmf.value IS NOT NULL
        AND bmf.metric_name = 'poke_battle_win'
    GROUP BY ji.episode_id, team_id
),
aggregated AS (
    SELECT 
        team_id,
        avg(value) as mean_value,
        count(*) as sample_size,
        stddevPop(value) as stddev,
        1.96 * stddevPop(value) / sqrt(count(*)) as confidence_interval
    FROM unique_feedback
    GROUP BY team_id
)
SELECT 
    team_id,
    mean_value,
    sample_size,
    mean_value - confidence_interval as ci_lower,
    mean_value + confidence_interval as ci_upper
FROM aggregated
ORDER BY mean_value DESC