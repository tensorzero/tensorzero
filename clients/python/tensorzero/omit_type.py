"""
Helper module for the `OMIT` sentinel value in generated dataclasses, to distinguish between omitted fields and
values that are explicitly set to None in update APIs.
"""


class OmitType:
    """Sentinel value to distinguish between omitted fields and values that are explicitly set to None."""

    def __repr__(self):
        return "OMIT"


OMIT = OmitType()
"""
Sentinel value to distinguish between omitted and null in API requests.

Usage:
- OMIT: Field is omitted (don't change existing value)
- None: Field is explicitly set to null
- value: Field is set to the provided value

Example:
    # Omit the field entirely (don't update it)
    update = UpdateRequest(name=OMIT)

    # Set the field to null (clear the existing value)
    update = UpdateRequest(name=None)

    # Set the field to a value
    update = UpdateRequest(name="new_value")
"""
