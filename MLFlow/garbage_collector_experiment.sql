DELETE FROM dbo.experiment_tags WHERE experiment_id in (
    SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
    );
DELETE FROM dbo.latest_metrics WHERE run_uuid in (
    SELECT run_uuid FROM dbo.runs WHERE experiment_id in (
        SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
    )
);
DELETE FROM dbo.metrics WHERE run_uuid in (
    SELECT run_uuid FROM dbo.runs WHERE experiment_id in (
        SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
    )
);
DELETE FROM dbo.tags WHERE run_uuid in (
    SELECT run_uuid FROM dbo.runs WHERE experiment_id in (
        SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
    )
);
DELETE FROM dbo.params WHERE run_uuid in (
    SELECT run_uuid FROM dbo.runs where experiment_id in (
        SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
));
DELETE FROM dbo.runs WHERE experiment_id in (
    SELECT experiment_id FROM dbo.experiments where lifecycle_stage='deleted'
);
DELETE FROM dbo.experiments where lifecycle_stage='deleted';

DELETE FROM dbo.datasets WHERE experiment_id=ANY(
SELECT experiment_id FROM experiments where lifecycle_stage='deleted');