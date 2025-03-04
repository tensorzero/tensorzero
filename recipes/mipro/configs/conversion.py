from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, create_model


def create_pydantic_model_from_schema(schema_dict: Dict[str, Any]) -> Type[BaseModel]:
    """Convert a JSON schema dictionary into a Pydantic BaseModel dynamically."""
    properties = schema_dict.get("properties", {})
    required_fields = set(schema_dict.get("required", []))

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def resolve_field_type(field_schema: Dict[str, Any]) -> Any:
        field_type = field_schema.get("type")
        if isinstance(field_type, list):
            # Filter out 'null' to check for non-null types.
            non_null_types = [
                type_mapping.get(t, Any) for t in field_type if t != "null"
            ]
            if "null" in field_type:
                # If only one non-null type, use Optional; otherwise, use Union inside Optional.
                if len(non_null_types) == 1:
                    return Optional[non_null_types[0]]
                else:
                    return Optional[Union[tuple(non_null_types)]]
            else:
                # If no 'null', use Union if there's more than one type.
                if len(non_null_types) == 1:
                    return non_null_types[0]
                else:
                    return Union[tuple(non_null_types)]
        else:
            return type_mapping.get(field_type, Any)

    model_fields = {
        key: (resolve_field_type(value), ...)
        if key in required_fields
        else (Optional[resolve_field_type(value)], None)
        for key, value in properties.items()
    }

    return create_model(schema_dict.get("title", "DynamicModel"), **model_fields)
