from tensorzero import ChatDatapointInsert, JsonDatapointInsert, TensorZeroGateway

# You can set up datapoints using dicts ...
extract_recipient_datapoint = {
    "function_name": "extract_recipient",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please send this to Alice at alice@example.com",
                    }
                ],
            }
        ]
    },
}

# ... or using the types `JsonDatapointInsert` ...
extract_recipient_datapoint_with_output = JsonDatapointInsert(
    function_name="extract_recipient",
    input={
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please send this to Bob at bob@example.com",
                    }
                ],
            }
        ]
    },
    output={
        "name": "Bob",
        "email": "bob@example.com",
    },
    name="bob_recipient_example",
)

# ... and `ChatDatapointInsert`.
draft_email_datapoint = ChatDatapointInsert(
    function_name="draft_email",
    input={
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please draft an email to Bob at bob@example.com",
                    }
                ],
            }
        ]
    },
    tags={
        "customer_id": "123",
    },
)


with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    # Insert datapoints
    insert_datapoints_response = t0.insert_datapoints(
        dataset_name="email_application",
        datapoints=[
            extract_recipient_datapoint,
            extract_recipient_datapoint_with_output,
            draft_email_datapoint,
        ],
    )

    print("insert_datapoints_response:\n")
    print(insert_datapoints_response)

    # Retrieve a single datapoint
    get_datapoint_response = t0.get_datapoint(
        dataset_name="email_application",
        datapoint_id=insert_datapoints_response[0],
    )

    print("\nget_datapoint_response:\n")
    print(get_datapoint_response)

    # Delete a single datapoint
    t0.delete_datapoint(
        dataset_name="email_application",
        datapoint_id=insert_datapoints_response[0],
    )

    print("\ndelete_datapoint_response:\n")
    print("N/A")

    # List datapoints
    list_datapoints_response = t0.list_datapoints(dataset_name="email_application")

    print("\nlist_datapoints_response:\n")
    print(list_datapoints_response)

    # Clean up (delete the remaining datapoints)
    for datapoint in list_datapoints_response:
        t0.delete_datapoint(
            dataset_name="email_application",
            datapoint_id=datapoint.id,
        )
