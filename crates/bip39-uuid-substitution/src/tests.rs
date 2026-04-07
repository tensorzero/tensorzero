use super::*;

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use std::error::Error;
use tensorzero_core::endpoints::inference::JsonInferenceResponse;
use tensorzero_core::inference::types::{
    Arguments, File, FinishReason, JsonInferenceOutput, Template, Thought, ThoughtSummaryBlock,
    Unknown, Usage,
};
use tensorzero_core::tool::{InferenceResponseToolCall, ToolCall, ToolCallWrapper, ToolResult};
use uuid::Uuid;

use durable_tools::Role;

fn make_uuid(value: u128) -> Uuid {
    Uuid::from_u128(value)
}

#[test]
fn test_uuid_to_triple_deterministic() {
    let uuid = make_uuid(0x550e_8400_e29b_41d4_a716_4466_5544_0000_u128);
    let mut substituter = UuidSubstituter::new();
    let first = substituter.substitute_uuids(&uuid.to_string()).unwrap();
    let second = substituter.substitute_uuids(&uuid.to_string()).unwrap();
    assert_eq!(first, second);
    assert_eq!(substituter.len(), 1);
}

#[test]
fn test_round_trip() {
    let uuid = make_uuid(42);
    let mut substituter = UuidSubstituter::new();
    let triple = substituter.substitute_uuids(&uuid.to_string()).unwrap();
    let round_trip = substituter.substitute_triples(&triple).unwrap();
    assert_eq!(round_trip, uuid.to_string());
}

#[test]
fn test_multiple_uuids_in_string() {
    let uuid1 = make_uuid(1);
    let uuid2 = make_uuid(2);
    let text = format!("{uuid1} and {uuid2}");
    let mut sub = UuidSubstituter::new();
    let substituted = sub.substitute_uuids(&text).unwrap();
    let round_trip = sub.substitute_triples(&substituted).unwrap();
    assert_eq!(round_trip, text);
}

#[test]
fn test_uuid_in_json() {
    let uuid = make_uuid(7);
    let mut value = json!({
        "outer": [uuid.to_string(), {"inner": uuid.to_string()}],
        "number": 1
    });
    let original = value.clone();
    let mut substituter = UuidSubstituter::new();
    process_text_in_value(&mut value, &mut |s| substituter.substitute_uuids(s)).unwrap();
    process_text_in_value(&mut value, &mut |s| substituter.substitute_triples(s)).unwrap();
    assert_eq!(value, original);
}

#[test]
fn test_unknown_triple_ignored() {
    let substituter = UuidSubstituter::new();
    let triple = "abandon-ability-able";
    let result = substituter.substitute_triples(triple).unwrap();
    assert_eq!(result, triple);
}

#[test]
fn test_tool_call_id_unchanged() {
    let id_uuid = make_uuid(100);
    let arg_uuid = make_uuid(200);
    let tool_call = ToolCall {
        id: id_uuid.to_string(),
        name: "do_thing".to_string(),
        arguments: arg_uuid.to_string(),
    };
    let message = InputMessage {
        role: Role::Assistant,
        content: vec![InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(
            tool_call,
        ))],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();

    let content = &processed.content[0];
    assert!(matches!(
        content,
        InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(_))
    ));
    if let InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(call)) = content {
        assert_eq!(call.id, id_uuid.to_string());
        let round_trip = substituter.substitute_triples(&call.arguments).unwrap();
        assert_eq!(round_trip, arg_uuid.to_string());
    }
}

#[test]
fn test_tool_result_id_unchanged() {
    let id_uuid = make_uuid(300);
    let result_uuid = make_uuid(400);
    let tool_result = ToolResult {
        id: id_uuid.to_string(),
        name: "do_thing".to_string(),
        result: result_uuid.to_string(),
    };
    let message = InputMessage {
        role: Role::Assistant,
        content: vec![InputMessageContent::ToolResult(tool_result)],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();

    let content = &processed.content[0];
    assert!(matches!(content, InputMessageContent::ToolResult(_)));
    if let InputMessageContent::ToolResult(result) = content {
        assert_eq!(result.id, id_uuid.to_string());
        let round_trip = substituter.substitute_triples(&result.result).unwrap();
        assert_eq!(round_trip, result_uuid.to_string());
    }
}

#[test]
fn test_file_block_unchanged() -> Result<(), Box<dyn Error>> {
    let file: File = serde_json::from_value(json!({
        "file_type": "url",
        "url": "https://example.com/file.txt"
    }))?;
    let message = InputMessage {
        role: Role::User,
        content: vec![InputMessageContent::File(file.clone())],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();
    assert_eq!(processed.content, vec![InputMessageContent::File(file)]);
    Ok(())
}

#[test]
fn test_unknown_block_unchanged() {
    let uuid = make_uuid(500);
    let unknown = Unknown {
        data: json!({"id": uuid.to_string()}),
        model_name: None,
        provider_name: None,
    };
    let message = InputMessage {
        role: Role::User,
        content: vec![InputMessageContent::Unknown(unknown.clone())],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();
    assert_eq!(
        processed.content,
        vec![InputMessageContent::Unknown(unknown)]
    );
}

#[test]
fn test_template_name_unchanged() {
    let name_uuid = make_uuid(600);
    let arg_uuid = make_uuid(700);
    let mut args = Map::new();
    args.insert("id".to_string(), Value::String(arg_uuid.to_string()));
    let template = Template {
        name: name_uuid.to_string(),
        arguments: Arguments(args),
    };
    let message = InputMessage {
        role: Role::User,
        content: vec![InputMessageContent::Template(template)],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();

    let content = &processed.content[0];
    assert!(matches!(content, InputMessageContent::Template(_)));
    if let InputMessageContent::Template(template) = content {
        assert_eq!(template.name, name_uuid.to_string());
        let value = template.arguments.0.get("id");
        assert!(matches!(value, Some(Value::String(_))));
        if let Some(Value::String(value)) = value {
            let round_trip = substituter.substitute_triples(value).unwrap();
            assert_eq!(round_trip, arg_uuid.to_string());
        }
    }
}

#[test]
fn test_thought_signature_unchanged() {
    let text_uuid = make_uuid(800);
    let signature_uuid = make_uuid(900);
    let summary_uuid = make_uuid(1000);
    let thought = Thought {
        text: Some(text_uuid.to_string()),
        signature: Some(signature_uuid.to_string()),
        summary: Some(vec![ThoughtSummaryBlock::SummaryText {
            text: summary_uuid.to_string(),
        }]),
        provider_type: None,
        extra_data: None,
    };
    let message = InputMessage {
        role: Role::Assistant,
        content: vec![InputMessageContent::Thought(thought)],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();

    let content = &processed.content[0];
    assert!(matches!(content, InputMessageContent::Thought(_)));
    if let InputMessageContent::Thought(thought) = content {
        let text = thought.text.as_ref();
        assert!(text.is_some());
        if let Some(text) = text {
            let round_trip = substituter.substitute_triples(text).unwrap();
            assert_eq!(round_trip, text_uuid.to_string());
        }
        assert_eq!(thought.signature, Some(signature_uuid.to_string()));
        assert_eq!(
            thought.summary,
            Some(vec![ThoughtSummaryBlock::SummaryText {
                text: summary_uuid.to_string()
            }])
        );
    }
}

proptest! {
    #[test]
    fn prop_round_trip_any_uuid(bytes in any::<[u8; 16]>()) {
        let uuid = Uuid::from_bytes(bytes);
        let mut substituter = UuidSubstituter::new();
        let triple = substituter.substitute_uuids(&uuid.to_string()).unwrap();
        let round_trip = substituter.substitute_triples(&triple).unwrap();
        prop_assert_eq!(round_trip, uuid.to_string());
    }

    #[test]
    fn prop_multiple_uuids_round_trip(bytes in proptest::collection::vec(any::<[u8; 16]>(), 1..8)) {
        let uuids: Vec<Uuid> = bytes.into_iter().map(Uuid::from_bytes).collect();
        let text = uuids
            .iter()
            .map(Uuid::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        let mut substituter = UuidSubstituter::new();
        let substituted = substituter.substitute_uuids(&text).unwrap();
        let round_trip = substituter.substitute_triples(&substituted).unwrap();
        prop_assert_eq!(round_trip, text);
    }

    #[test]
    fn prop_json_round_trip(bytes in any::<[u8; 16]>(), bytes2 in any::<[u8; 16]>()) {
        let uuid1 = Uuid::from_bytes(bytes);
        let uuid2 = Uuid::from_bytes(bytes2);
        let mut value = json!({
            "a": uuid1.to_string(),
            "b": [uuid2.to_string(), {"c": uuid1.to_string()}],
            "d": 12
        });
        let original = value.clone();
        let mut substituter = UuidSubstituter::new();
        process_text_in_value(&mut value, &mut |s| substituter.substitute_uuids(s)).unwrap();
        process_text_in_value(&mut value, &mut |s| substituter.substitute_triples(s)).unwrap();
        prop_assert_eq!(value, original);
    }

    #[test]
    fn prop_deterministic_generation(bytes in any::<[u8; 16]>()) {
        let uuid = Uuid::from_bytes(bytes);
        let mut substituter = UuidSubstituter::new();
        let first = substituter.substitute_uuids(&uuid.to_string()).unwrap();
        let second = substituter.substitute_uuids(&uuid.to_string()).unwrap();
        prop_assert_eq!(first, second);
    }
}

#[test]
fn test_tool_call_wrapper_arguments() {
    let raw_uuid = make_uuid(1100);
    let parsed_uuid = make_uuid(1200);
    let tool_call = InferenceResponseToolCall {
        id: "call-1".to_string(),
        raw_name: "tool".to_string(),
        raw_arguments: raw_uuid.to_string(),
        name: Some("tool".to_string()),
        arguments: Some(json!({"id": parsed_uuid.to_string()})),
    };
    let message = InputMessage {
        role: Role::Assistant,
        content: vec![InputMessageContent::ToolCall(
            ToolCallWrapper::InferenceResponseToolCall(tool_call),
        )],
    };
    let mut substituter = UuidSubstituter::new();
    let processed = preprocess_message(&mut substituter, message).unwrap();

    let content = &processed.content[0];
    assert!(matches!(
        content,
        InputMessageContent::ToolCall(ToolCallWrapper::InferenceResponseToolCall(_))
    ));
    if let InputMessageContent::ToolCall(ToolCallWrapper::InferenceResponseToolCall(tool_call)) =
        content
    {
        let raw_round_trip = substituter
            .substitute_triples(&tool_call.raw_arguments)
            .unwrap();
        assert_eq!(raw_round_trip, raw_uuid.to_string());
        let arguments = tool_call.arguments.as_ref();
        assert!(matches!(arguments, Some(Value::Object(_))));
        if let Some(Value::Object(map)) = arguments {
            let value = map.get("id");
            assert!(matches!(value, Some(Value::String(_))));
            if let Some(Value::String(value)) = value {
                let parsed_round_trip = substituter.substitute_triples(value).unwrap();
                assert_eq!(parsed_round_trip, parsed_uuid.to_string());
            }
        }
    }
}

#[test]
fn test_collision_error() {
    let uuid1 = make_uuid(1);
    let uuid2 = make_uuid(2);
    let mut substituter = UuidSubstituter::new();

    // Generate the triple that uuid2 would produce
    let triple_for_uuid2 = generate_triple(&uuid2);

    // Inject a mapping so that uuid1 maps to the same triple as uuid2 would
    substituter.inject_mapping(uuid1, triple_for_uuid2.clone());

    // Now when we try to substitute uuid2, it should detect the collision
    let result = substituter.substitute_uuids(&uuid2.to_string());
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("UUID collision"));
    match err {
        UuidSubstitutionError::Collision {
            new_uuid,
            existing_uuid,
            triple,
        } => {
            assert_eq!(new_uuid, uuid2);
            assert_eq!(existing_uuid, uuid1);
            assert_eq!(triple, triple_for_uuid2);
        }
        UuidSubstitutionError::MissingCaptureGroup { .. } => {
            panic!("Expected Collision error")
        }
    }
}

#[test]
fn test_case_insensitive_triple_matching() {
    let uuid = make_uuid(42);
    let mut substituter = UuidSubstituter::new();

    // Get the lowercase triple for the UUID
    let triple = substituter.substitute_uuids(&uuid.to_string()).unwrap();

    // Verify it's lowercase
    assert_eq!(triple, triple.to_lowercase());

    // Test various case variations
    let uppercase = triple.to_uppercase();
    let mixed_case = triple
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i % 2 == 0 {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect::<String>();

    // All case variations should resolve back to the original UUID
    assert_eq!(
        substituter.substitute_triples(&triple).unwrap(),
        uuid.to_string()
    );
    assert_eq!(
        substituter.substitute_triples(&uppercase).unwrap(),
        uuid.to_string()
    );
    assert_eq!(
        substituter.substitute_triples(&mixed_case).unwrap(),
        uuid.to_string()
    );
}

#[test]
fn test_postprocess_json_response() {
    let uuid1 = make_uuid(1300);
    let uuid2 = make_uuid(1400);

    // Create a substituter and register the UUIDs by substituting them
    let mut substituter = UuidSubstituter::new();
    let triple1 = substituter.substitute_uuids(&uuid1.to_string()).unwrap();
    let triple2 = substituter.substitute_uuids(&uuid2.to_string()).unwrap();

    // Create a JSON inference response with triples in both raw and parsed fields
    let raw_json = format!(r#"{{"id": "{triple1}", "ref": "{triple2}"}}"#);
    let parsed_json = json!({"id": triple1, "ref": triple2});

    let json_response = InferenceResponse::Json(JsonInferenceResponse {
        inference_id: Uuid::nil(),
        episode_id: Uuid::nil(),
        variant_name: "test".to_string(),
        output: JsonInferenceOutput {
            raw: Some(raw_json),
            parsed: Some(parsed_json),
        },
        usage: Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cost: None,
            provider_cache_read_input_tokens: None,
            provider_cache_write_input_tokens: None,
        },
        raw_usage: None,
        original_response: None,
        raw_response: None,
        finish_reason: Some(FinishReason::Stop),
    });

    // Postprocess the response
    let processed = postprocess_response(&substituter, json_response).unwrap();

    // Verify the triples were converted back to UUIDs
    if let InferenceResponse::Json(json) = processed {
        let expected_raw = format!(r#"{{"id": "{uuid1}", "ref": "{uuid2}"}}"#);
        assert_eq!(json.output.raw, Some(expected_raw));

        let expected_parsed = json!({"id": uuid1.to_string(), "ref": uuid2.to_string()});
        assert_eq!(json.output.parsed, Some(expected_parsed));
    } else {
        panic!("Expected Json response");
    }
}
