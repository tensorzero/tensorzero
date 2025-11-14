#!/usr/bin/env python3
"""
Generate feedback fixture data with timestamps spread across multiple days for extract_entities.

This creates jaro_winkler_similarity feedback for the extract_entities function's
track_and_stop candidate variants (gpt4o_mini_initial_prompt, gpt4o_initial_prompt,
llama_8b_initial_prompt) with timestamps distributed across different days and hours
to enable testing the feedback timeseries chart at different time granularities.
"""

import json
import random
import uuid
from datetime import datetime, timedelta

# Sample inference IDs for each variant (from the database query)
INFERENCE_IDS = {
    "gpt4o_mini_initial_prompt": [
        "0196368f-1aec-7ea3-80a4-8831d35fb64a",
        "0196368f-1aec-7ea3-80a4-885eb2663b5c",
        "0196368f-1aec-7ea3-80a4-8873c4726d26",
        "0196368f-1aec-7ea3-80a4-88931ffe91e1",
        "01943767-b110-74e2-ab47-f427246eb993",
        "0196374c-2c5c-7d50-8183-586eb91b4f12",
        "0196374c-2c5c-7d50-8183-5888c02cc15f",
        "0196374c-2c5c-7d50-8183-58a06b3f301c",
        "0196374c-2c5c-7d50-8183-58cf4c1a5c2f",
        "01943830-0e08-7c81-8909-a7521de244f1",
    ],
    "gpt4o_initial_prompt": [
        "01939adf-0f50-79d0-8d55-7a009fcc5e32",
        "01939ba7-6c48-7492-a810-ea6f4d5c4c77",
        "01939c6f-c940-7290-8686-fc2bf81e32ba",
        "01939d38-2638-7aa1-99d1-a644144b23ae",
        "01939e00-8330-7913-b31d-b20feed25a31",
        "01939ec8-e028-7ee2-9136-23d50a7aace2",
        "01939f91-3d20-7f43-8fc8-0c8d822e14ae",
        "0193a059-9a18-7881-8b31-5676eb320e7d",
        "0193a121-f710-7fa2-9024-e6eeed6c103d",
        "0193a1ea-5408-7982-8c08-85417a48cf20",
    ],
    "llama_8b_initial_prompt": [
        "0193e14f-be80-7a42-8692-49469910a058",
        "0196367b-1ca4-7cc2-80d5-47adaed50feb",
        "0196367b-1c9b-7d63-81cf-ec4db8e3b5f7",
        "0193e218-1b78-76b0-a1d2-5ba65b6ee00d",
        "0193e2e0-7870-7582-acbe-f4fc60eccab4",
        "0193e3a8-d568-70f2-8e17-1c6f64ce05a7",
        "0193e471-3260-7cb1-9d2a-662294d41162",
        "0193e539-8f58-77c3-916f-7e82b43e19c1",
        "0196367b-202e-7092-83f5-b649b7a1aba9",
        "0193e601-ec50-7912-a06c-58a985bf82e3",
    ],
}


def datetime_to_uuidv7_hex(dt: datetime) -> str:
    """
    Generate a UUIDv7 hex string with the given datetime.
    UUIDv7 format: unix_ts_ms (48 bits) | version (4 bits) | rand_a (12 bits) |
                   variant (2 bits) | rand_b (62 bits)
    """
    # Convert datetime to milliseconds since epoch
    unix_ts_ms = int(dt.timestamp() * 1000)

    # Generate random bits
    rand_a = random.getrandbits(12)
    rand_b = random.getrandbits(62)

    # Construct UUID
    # First 48 bits: timestamp
    uuid_int = unix_ts_ms << 80
    # Next 4 bits: version (0111 for UUIDv7)
    uuid_int |= 0x7 << 76
    # Next 12 bits: random_a
    uuid_int |= rand_a << 64
    # Next 2 bits: variant (10)
    uuid_int |= 0x2 << 62
    # Last 62 bits: random_b
    uuid_int |= rand_b

    # Convert to UUID string
    uuid_obj = uuid.UUID(int=uuid_int)
    return str(uuid_obj)


def generate_feedback_entries():
    """
    Generate feedback entries spread across multiple days and hours.

    Distribution strategy:
    - Spread across 7 days (to test week, day, hour granularities)
    - Multiple entries per day (to test daily aggregation)
    - Multiple entries per hour (to test hourly aggregation)
    - 10-15 samples per variant per day
    """

    # Start date: 7 days ago from now
    base_date = datetime.now() - timedelta(days=7)

    feedback_entries = []

    # Generate feedback for 7 days
    for day_offset in range(7):
        current_day = base_date + timedelta(days=day_offset)

        for variant_name, inference_ids in INFERENCE_IDS.items():
            # Generate 10-15 samples per variant per day
            samples_per_day = random.randint(10, 15)

            for sample_idx in range(samples_per_day):
                # Spread throughout the day (different hours)
                hour_offset = random.randint(0, 23)
                minute_offset = random.randint(0, 59)
                second_offset = random.randint(0, 59)

                feedback_time = current_day + timedelta(hours=hour_offset, minutes=minute_offset, seconds=second_offset)

                # Generate UUIDv7 with this timestamp
                feedback_id = datetime_to_uuidv7_hex(feedback_time)

                # Pick a random inference ID for this variant
                target_id = random.choice(inference_ids)

                # Generate a realistic jaro_winkler_similarity value (0.0 to 1.0)
                value = round(random.uniform(0.0, 1.0), 6)

                entry = {
                    "id": feedback_id,
                    "target_id": target_id,
                    "metric_name": "jaro_winkler_similarity",
                    "value": value,
                    "tags": {},
                }

                feedback_entries.append(entry)

    return feedback_entries


def main():
    """Generate and output feedback entries as JSONL."""
    entries = generate_feedback_entries()

    # Sort by timestamp (encoded in UUIDv7)
    entries.sort(key=lambda e: e["id"])

    # Print statistics
    print(f"# Generated {len(entries)} feedback entries", file=open("/dev/stderr", "w"))
    print(f"# Variants: {list(INFERENCE_IDS.keys())}", file=open("/dev/stderr", "w"))
    print("# Time span: 7 days", file=open("/dev/stderr", "w"))

    # Output JSONL
    for entry in entries:
        print(json.dumps(entry))


if __name__ == "__main__":
    main()
