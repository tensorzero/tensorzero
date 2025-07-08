import json
import os
from time import sleep
from typing import Any, Dict
from uuid import UUID

import pandas as pd
from clickhouse_connect import get_client  # type: ignore
from tensorzero.util import uuid7


def test_double_feedback_query() -> None:
    """
    Most fine-tuning recipes will use a query that pulls all inference for a function and the
    feedback associated with that inference, if the feedback is above / below a certain threshold.

    We want to drop rows for inferences or episodes where there are multiple feedbacks corresponding
    to the inference and only take the most recent one.
    This test checks that the query works as expected.
    """

    assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, (
        "TENSORZERO_CLICKHOUSE_URL environment variable not set"
    )
    client = get_client(dsn=os.environ["TENSORZERO_CLICKHOUSE_URL"])

    # Insert an Inference we can use to assign feedback to
    inference: Dict[str, Any] = {
        "id": str(uuid7()),
        "function_name": "test_function",
        "variant_name": "test_variant",
        "episode_id": str(uuid7()),
        "input": "test_input",
        "output": "test_output",
        "inference_parameters": {},
        "processing_time_ms": 100,
    }

    query: str = f"""
    INSERT INTO ChatInference
    FORMAT JSONEachRow
    {json.dumps(inference)}
    """

    client.query(query)  # type: ignore

    # Insert a float feedback for the inference
    first_feedback: Dict[str, Any] = {
        "target_id": inference["id"],
        "value": 1.0,
        "metric_name": "test_metric",
        "id": str(uuid7()),
    }

    query: str = f"""
    INSERT INTO FloatMetricFeedback
    FORMAT JSONEachRow
    {json.dumps(first_feedback)}
    """

    client.query(query)  # type: ignore

    # Sleep to ensure the first feedback is recorded with a different timestamp
    # Since the granularity of the timestamp is 1 second, we need to sleep for more than 1 second
    sleep(1.1)

    # Insert a second feedback for the inference
    second_feedback: Dict[str, Any] = {
        "target_id": inference["id"],
        "value": 4.0,
        "metric_name": "test_metric",
        "id": str(uuid7()),
    }

    query: str = f"""
    INSERT INTO FloatMetricFeedback
    FORMAT JSONEachRow
    {json.dumps(second_feedback)}
    """

    client.query(query)  # type: ignore

    # At this point we have a duplicate feedback for the inference. Let's test that the query we use in the recipes gets the later one.
    query: str = """
    SELECT
        i.variant_name,
        i.input,
        i.output,
        f.value,
        i.episode_id
    FROM
        ChatInference i
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

    # NOTE: we add the last AND i.id = %(inference_id)s to ensure that the query is using the inference id we set above
    # and we can run the test multiple times without it failing
    result: pd.DataFrame = client.query_df(query, {"inference_id": inference["id"]})  # type: ignore
    first_row = result.iloc[0]  # type: ignore

    assert len(result) == 1  # type: ignore
    assert first_row["variant_name"] == "test_variant"
    assert first_row["input"] == "test_input"
    assert first_row["output"] == "test_output"
    assert first_row["value"] == 4.0
    assert first_row["episode_id"] == UUID(str(inference["episode_id"]))
