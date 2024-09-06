import json
import os
from uuid import UUID

from clickhouse_driver import Client
from uuid_extensions import uuid7


def test_double_feedback_query():
    """
    Most fine-tuning recipes will use a query that pulls all inference for a function and the
    feedback associated with that inference, if the feedback is above / below a certain threshold.

    We want to drop rows for inferences or episodes where there are multiple feedbacks corresponding
    to the inference and only take the most recent one.
    This test checks that the query works as expected.
    """

    CLICKHOUSE_NATIVE_URL = os.getenv("CLICKHOUSE_NATIVE_URL")
    client = Client.from_url(CLICKHOUSE_NATIVE_URL)
    # Insert an Inference we can use to assign feedback to
    inference = {
        "id": str(uuid7()),
        "function_name": "test_function",
        "variant_name": "test_variant",
        "episode_id": str(uuid7()),
        "input": "test_input",
        "output": "test_output",
        "inference_parameters": {},
        "processing_time_ms": 100,
    }
    query = f"""
    INSERT INTO Inference
    FORMAT JSONEachRow
    {json.dumps(inference)}
    """
    client.execute(query)

    # Insert a float feedback for the inference
    first_feedback = {
        "target_id": inference["id"],
        "value": 1.0,
        "metric_name": "test_metric",
        "id": str(uuid7()),
    }
    query = f"""
    INSERT INTO FloatMetricFeedback
    FORMAT JSONEachRow
    {json.dumps(first_feedback)}
    """
    client.execute(query)

    # Insert a second feedback for the inference
    second_feedback = {
        "target_id": inference["id"],
        "value": 4.0,
        "metric_name": "test_metric",
        "id": str(uuid7()),
    }
    query = f"""
    INSERT INTO FloatMetricFeedback
    FORMAT JSONEachRow
    {json.dumps(second_feedback)}
    """
    client.execute(query)

    # At this point we have a duplicate feedback for the inference. Let's test that the query we use in the recipes gets the later one.
    query = """
    SELECT
        i.variant_name,
        i.input,
        i.output,
        f.value,
        i.episode_id
    FROM
        Inference i
    JOIN
        (SELECT
            target_id,
            value,
            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
        FROM
            FloatMetricFeedback
        WHERE
            metric_name = 'test_metric'
            AND value > 0.5
        ) f ON i.id = f.target_id AND f.rn = 1
    WHERE
        i.function_name = 'test_function'
        AND i.id = %(inference_id)s
    """
    result = client.execute(query, params={"inference_id": inference["id"]})
    assert len(result) == 1
    assert result[0] == (
        "test_variant",
        "test_input",
        "test_output",
        4.0,
        UUID(inference["episode_id"]),
    )
