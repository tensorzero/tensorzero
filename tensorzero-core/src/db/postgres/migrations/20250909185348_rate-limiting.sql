-- Set up resource_bucket table
CREATE TABLE resource_bucket (
    key TEXT PRIMARY KEY,
    tickets BIGINT NOT NULL CHECK (tickets >= 0),
    balance_as_of TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- For returning the up-to-date state of a bucket
CREATE TYPE calculated_bucket_state AS (
    available_tickets BIGINT,
    new_balance_as_of TIMESTAMPTZ,
    intervals_passed  BIGINT
);

-- Internal helper function that calculates the up-to-date state of a bucket
-- prior to any modifications
CREATE OR REPLACE FUNCTION _calculate_refilled_state(
    p_current_tickets BIGINT,
    p_balance_as_of TIMESTAMPTZ,
    p_capacity BIGINT,
    p_refill_amount BIGINT,
    p_refill_interval INTERVAL
)
RETURNS calculated_bucket_state
LANGUAGE plpgsql AS $$
DECLARE
    v_intervals_passed BIGINT;
    v_available_tickets BIGINT;
    v_result calculated_bucket_state;
BEGIN
    -- Calculate how many whole refill intervals have passed.
    IF p_refill_interval > '0 seconds'::interval THEN
        -- EPOCH FROM converts to a number of seconds
        -- Allows us to do integer division rather than interval division
        v_intervals_passed := floor(
            EXTRACT(EPOCH FROM (now() - p_balance_as_of)) / EXTRACT(EPOCH FROM p_refill_interval)
        );
    ELSE
        v_intervals_passed := 0;
    END IF;

    -- Ensure we don't go backwards in time
    -- (e.g., due to transaction time < p_balance_as_of after a race)
    v_intervals_passed := GREATEST(v_intervals_passed, 0);

    -- Calculate available tickets: current + refilled, capped at capacity.
    v_available_tickets := least(
        p_current_tickets + (v_intervals_passed * p_refill_amount),
        p_capacity
    );

    -- Build the result using our custom type
    v_result.available_tickets := v_available_tickets;
    v_result.intervals_passed := v_intervals_passed;
    v_result.new_balance_as_of := p_balance_as_of + (v_intervals_passed * p_refill_interval);

    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION consume_resource_tickets(
    p_key TEXT,
    p_requested BIGINT,
    p_capacity BIGINT,
    p_refill_amount BIGINT,
    p_refill_interval INTERVAL
)
RETURNS TABLE (success BOOLEAN, tickets_remaining BIGINT, tickets_consumed BIGINT)
LANGUAGE plpgsql AS $$
DECLARE
    bucket_state record;
    refilled_state calculated_bucket_state;
    is_successful boolean;
BEGIN
    -- Step 1: Atomically get-or-create the bucket, locking the row.
    INSERT INTO resource_bucket (key, tickets, balance_as_of)
    VALUES (p_key, p_capacity, now())
    ON CONFLICT (key) DO UPDATE SET key = EXCLUDED.key
    RETURNING * INTO bucket_state;

    -- Step 2: Calculate the refilled state using the helper function.
    SELECT * INTO refilled_state
    FROM _calculate_refilled_state(
        bucket_state.tickets,
        bucket_state.balance_as_of,
        p_capacity,
        p_refill_amount,
        p_refill_interval
    );

    -- Step 3: Determine if the consumption is successful.
    is_successful := refilled_state.available_tickets >= p_requested;

    IF is_successful THEN
        tickets_remaining := refilled_state.available_tickets - p_requested;
        tickets_consumed := p_requested;
    ELSE
        tickets_remaining := refilled_state.available_tickets;
        tickets_consumed := 0;
    END IF;

    -- Step 4: Update the bucket with the new state.
    UPDATE resource_bucket
    SET
        tickets = tickets_remaining,
        balance_as_of = refilled_state.new_balance_as_of
    WHERE key = p_key;

    -- Step 5: Return the result.
    RETURN QUERY SELECT is_successful, tickets_remaining, tickets_consumed;
END;
$$;

CREATE OR REPLACE FUNCTION return_resource_tickets(
    p_key TEXT,
    p_amount BIGINT,
    p_capacity BIGINT,
    p_refill_amount BIGINT,
    p_refill_interval INTERVAL
)
RETURNS TABLE (returned_tickets BIGINT)
LANGUAGE plpgsql AS $$
DECLARE
    bucket_state record;
    refilled_state calculated_bucket_state;
    final_balance BIGINT;
BEGIN
    -- Step 1: Atomically lock the row, creating it if it doesn't exist.
    INSERT INTO resource_bucket (key, tickets, balance_as_of)
    VALUES (p_key, p_capacity, now())
    ON CONFLICT (key) DO UPDATE SET key = EXCLUDED.key
    RETURNING * INTO bucket_state;

    -- Step 2: Calculate the refilled state using the helper function.
    SELECT * INTO refilled_state
    FROM _calculate_refilled_state(
        bucket_state.tickets,
        bucket_state.balance_as_of,
        p_capacity,
        p_refill_amount,
        p_refill_interval
    );

    -- Step 3: Calculate the new balance, capped at capacity.
    final_balance := least(
        refilled_state.available_tickets + p_amount,
        p_capacity
    );

    -- Step 4: Update the bucket with the new balance and timestamp.
    UPDATE resource_bucket
    SET
        tickets = final_balance,
        balance_as_of = refilled_state.new_balance_as_of
    WHERE key = p_key;

    -- Step 5: Return the result.
    RETURN QUERY SELECT final_balance;
END;
$$;
