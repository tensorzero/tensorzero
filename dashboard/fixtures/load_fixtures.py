import json
import os

import clickhouse_connect
import pandas as pd
from tensorzero.util import uuid7


def main():
    json_inference_df = pd.read_csv("json_inference_examples.csv")
    chat_inference_df = pd.read_csv("chat_inference_examples.csv")
    boolean_metric_feedback_df = pd.read_csv("boolean_metric_feedback_examples.csv")
    float_metric_feedback_df = pd.read_csv("float_metric_feedback_examples.csv")
    demonstration_feedback_df = pd.read_csv("demonstration_feedback_examples.csv")

    # Create ID mapping dictionaries
    json_inference_df["new_id"] = json_inference_df["id"].apply(lambda x: str(uuid7()))
    chat_inference_df["new_id"] = chat_inference_df["id"].apply(lambda x: str(uuid7()))

    json_id_mapping = dict(zip(json_inference_df["id"], json_inference_df["new_id"]))
    chat_id_mapping = dict(zip(chat_inference_df["id"], chat_inference_df["new_id"]))

    # Update target_ids in boolean metric feedback dataframe
    boolean_metric_feedback_df["target_id"] = (
        boolean_metric_feedback_df["target_id"]
        .map(json_id_mapping)
        .fillna(boolean_metric_feedback_df["target_id"])
    )
    boolean_metric_feedback_df["target_id"] = (
        boolean_metric_feedback_df["target_id"]
        .map(chat_id_mapping)
        .fillna(boolean_metric_feedback_df["target_id"])
    )

    # Update target_ids in float metric feedback dataframe
    float_metric_feedback_df["target_id"] = (
        float_metric_feedback_df["target_id"]
        .map(json_id_mapping)
        .fillna(float_metric_feedback_df["target_id"])
    )
    float_metric_feedback_df["target_id"] = (
        float_metric_feedback_df["target_id"]
        .map(chat_id_mapping)
        .fillna(float_metric_feedback_df["target_id"])
    )

    # Update target_ids in demonstration feedback dataframe
    demonstration_feedback_df["inference_id"] = (
        demonstration_feedback_df["inference_id"]
        .map(json_id_mapping)
        .fillna(demonstration_feedback_df["inference_id"])
    )
    demonstration_feedback_df["inference_id"] = (
        demonstration_feedback_df["inference_id"]
        .map(chat_id_mapping)
        .fillna(demonstration_feedback_df["inference_id"])
    )

    # Replace old IDs with new IDs
    json_inference_df["id"] = json_inference_df["new_id"]
    chat_inference_df["id"] = chat_inference_df["new_id"]

    # Drop the temporary new_id columns
    json_inference_df = json_inference_df.drop("new_id", axis=1)
    chat_inference_df = chat_inference_df.drop("new_id", axis=1)

    # Prep the dataframes for ClickHouse
    # `tags` columns need to be maps
    json_inference_df["tags"] = json_inference_df["tags"].apply(lambda x: json.loads(x))
    chat_inference_df["tags"] = chat_inference_df["tags"].apply(lambda x: json.loads(x))
    boolean_metric_feedback_df["tags"] = boolean_metric_feedback_df["tags"].apply(
        lambda x: json.loads(x)
    )
    float_metric_feedback_df["tags"] = float_metric_feedback_df["tags"].apply(
        lambda x: json.loads(x)
    )
    demonstration_feedback_df["tags"] = demonstration_feedback_df["tags"].apply(
        lambda x: json.loads(x)
    )
    # `tool_params` column needs to be a string
    chat_inference_df["tool_params"] = chat_inference_df["tool_params"].fillna("")

    # Get the ClickHouse client
    client = clickhouse_connect.get_client(dsn=os.environ["CLICKHOUSE_URL"])
    chat_inference_df["id"] = chat_inference_df["id"].astype(str)

    # Insert the dataframes into ClickHouse
    client.insert_df(df=json_inference_df, table="JsonInference")
    client.insert_df(df=chat_inference_df, table="ChatInference")
    client.insert_df(df=boolean_metric_feedback_df, table="BooleanMetricFeedback")
    client.insert_df(df=float_metric_feedback_df, table="FloatMetricFeedback")
    client.insert_df(df=demonstration_feedback_df, table="DemonstrationFeedback")

    print("Data inserted successfully. This process should exit with a 0 status code.")


if __name__ == "__main__":
    main()
