-- Dashboard queries for defect_inspection.inspection_events (Databricks SQL).
--
-- *** SIMULATED DATA ***  Every row comes from src/simulator.py: val images replayed at a simulated clock, with
-- drift injected on purpose. Title the dashboard "Defect inspection monitor - SIMULATED DATA" and prefix each panel
-- "[SIMULATED]". Each query filters `simulated = TRUE`, so real traffic (if it is ever logged) never mixes in.
--
-- Parameter :bucket_seconds (number, default 10). The simulated clock runs at ~5 events/s, so 10 s buckets give
-- ~50 events each. Timestamps are simulated time, not wall-clock time.
-- Statements are separated by "-- name:" lines; scripts can split on that marker.

-- name: 0_notice (counter / text panel)
SELECT 'SIMULATED DATA: synthetic traffic from val images with injected drift; not production' AS notice,
       COUNT(*)                          AS events,
       MIN(timestamp)                    AS first_event,
       MAX(timestamp)                    AS last_event,
       COUNT_IF(perturbation <> 'none')  AS perturbed_events
FROM defect_inspection.inspection_events
WHERE simulated = TRUE;

-- name: 1_events_over_time (bar)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       COUNT(*)                                      AS events,
       COUNT_IF(perturbation <> 'none')              AS perturbed_events
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY 1
ORDER BY 1;

-- name: 2_mean_max_confidence_by_bucket (line)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       ROUND(AVG(max_confidence), 4)                 AS mean_max_confidence,
       COUNT(*)                                      AS events
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY 1
ORDER BY 1;

-- name: 3_detection_count_by_bucket (line)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       ROUND(AVG(n_detections), 2)                   AS mean_detections_per_image,
       SUM(n_detections)                             AS total_detections
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY 1
ORDER BY 1;

-- name: 4_latency_percentiles_by_bucket (line; latency_ms = inference call only, measured on the simulator host)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       ROUND(percentile_approx(latency_ms, 0.50), 1) AS p50_ms,
       ROUND(percentile_approx(latency_ms, 0.95), 1) AS p95_ms,
       ROUND(percentile_approx(latency_ms, 0.99), 1) AS p99_ms
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY 1
ORDER BY 1;

-- name: 5a_per_action_detection_counts_by_bucket (stacked bar; counts all detections, incl. 'ignore')
SELECT bucket, stack(5, 'stop_now', stop_now, 'schedule_service', schedule_service, 'human_review', human_review,
                     'monitor', monitor, 'ignore', ignore_) AS (action, detections)
FROM (
  SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
         SUM(n_stop_now) AS stop_now, SUM(n_schedule_service) AS schedule_service,
         SUM(n_human_review) AS human_review, SUM(n_monitor) AS monitor, SUM(n_ignore) AS ignore_
  FROM defect_inspection.inspection_events
  WHERE simulated = TRUE
  GROUP BY 1
)
ORDER BY bucket, action;

-- name: 5b_worst_decision_per_image_by_bucket (stacked bar; one decision per image)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       worst_decision,
       COUNT(*) AS images
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: 6_input_statistics_by_bucket (four small line charts; the drift signal)
SELECT timestamp_seconds(CAST(FLOOR(unix_timestamp(timestamp) / :bucket_seconds) * :bucket_seconds AS BIGINT)) AS bucket,
       ROUND(AVG(brightness_mean), 2) AS brightness_mean,
       ROUND(AVG(contrast_std), 2)    AS contrast_std,
       ROUND(AVG(sharpness), 1)       AS sharpness,
       ROUND(AVG(noise_est), 2)       AS noise_est
FROM defect_inspection.inspection_events
WHERE simulated = TRUE AND schema_version >= 2
GROUP BY 1
ORDER BY 1;

-- name: 7_model_version (table / pie)
SELECT model_version,
       COUNT(*)       AS events,
       MIN(timestamp) AS first_seen,
       MAX(timestamp) AS last_seen
FROM defect_inspection.inspection_events
WHERE simulated = TRUE
GROUP BY model_version
ORDER BY events DESC;
