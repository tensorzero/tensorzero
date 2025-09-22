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
    IF p_refill_interval <= '0 seconds'::interval THEN
        RAISE EXCEPTION 'Refill interval must be positive, got: %', p_refill_interval;
    ELSE
        -- EPOCH FROM converts to a number of seconds
        -- Allows us to do integer division rather than interval division
        v_intervals_passed := floor(
            EXTRACT(EPOCH FROM (now() - p_balance_as_of)) / EXTRACT(EPOCH FROM p_refill_interval)
        );
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

-- Function to get the current balance without modification.
CREATE OR REPLACE FUNCTION get_resource_bucket_balance(
    p_key TEXT,
    p_capacity BIGINT,
    p_refill_amount BIGINT,
    p_refill_interval INTERVAL
)
RETURNS BIGINT
LANGUAGE plpgsql AS $$
DECLARE
    bucket_state record;
    refilled_state calculated_bucket_state;
BEGIN
    -- Step 1: Find the bucket.
    SELECT * INTO bucket_state FROM resource_bucket WHERE resource_bucket.key = p_key;

    -- Step 2: If the bucket doesn't exist, the balance is the capacity.
    IF NOT FOUND THEN
        RETURN p_capacity;
    END IF;

    -- Step 3: If the bucket exists, calculate its current refilled state.
    SELECT * INTO refilled_state
    FROM _calculate_refilled_state(
        bucket_state.tickets,
        bucket_state.balance_as_of,
        p_capacity,
        p_refill_amount,
        p_refill_interval
    );

    -- Step 4: Return the available tickets from the calculated state.
    RETURN refilled_state.available_tickets;
END;
$$;

-- Function to atomically consume tickets from multiple resource buckets.
CREATE OR REPLACE FUNCTION consume_multiple_resource_tickets(
    p_keys TEXT[],
    p_requested_amounts BIGINT[],
    p_capacities BIGINT[],
    p_refill_amounts BIGINT[],
    p_refill_intervals INTERVAL[]
)
RETURNS TABLE (bucket_key TEXT, is_successful BOOLEAN, tickets_remaining BIGINT, tickets_consumed BIGINT)
LANGUAGE plpgsql AS $$
DECLARE
    i INT;
    bucket_state RECORD;
    refilled_state calculated_bucket_state;
    failure_detected BOOLEAN := false;
    temp_row RECORD;
BEGIN
    -- Input validation: check for consistent array lengths
    IF array_length(p_keys, 1) IS NULL OR
       array_length(p_keys, 1) != array_length(p_requested_amounts, 1) OR
       array_length(p_keys, 1) != array_length(p_capacities, 1) OR
       array_length(p_keys, 1) != array_length(p_refill_amounts, 1) OR
       array_length(p_keys, 1) != array_length(p_refill_intervals, 1) THEN
        RAISE EXCEPTION 'Input arrays must have the same length';
    END IF;

    -- Input validation: check for duplicate keys
    IF array_length(p_keys, 1) > 0 AND array_length(p_keys, 1) != (
        SELECT COUNT(DISTINCT key_column)
        FROM unnest(p_keys) AS keys(key_column)
    ) THEN
        RAISE EXCEPTION 'Duplicate keys are not allowed in the input array';
    END IF;

    -- Create a temporary table to store intermediate results
    CREATE TEMP TABLE temp_bucket_states (
        key TEXT PRIMARY KEY,
        requested BIGINT,
        capacity BIGINT,
        refill_amount BIGINT,
        refill_interval INTERVAL,
        refilled_tickets BIGINT,
        new_balance_as_of TIMESTAMPTZ
    ) ON COMMIT DROP;

    -- Populate temp table from input arrays
    FOR i IN 1..array_length(p_keys, 1) LOOP
        INSERT INTO temp_bucket_states (key, requested, capacity, refill_amount, refill_interval)
        VALUES (p_keys[i], p_requested_amounts[i], p_capacities[i], p_refill_amounts[i], p_refill_intervals[i]);
    END LOOP;

    -- Pass 1: Lock rows in a consistent order, calculate new states, and check for sufficiency.
    FOR temp_row IN SELECT * FROM temp_bucket_states ORDER BY temp_bucket_states.key LOOP
        -- Atomically get-or-create the bucket, locking the row.
        INSERT INTO resource_bucket (key, tickets, balance_as_of)
        VALUES (temp_row.key, temp_row.capacity, now())
        ON CONFLICT (key) DO UPDATE SET key = EXCLUDED.key
        RETURNING * INTO bucket_state;

        -- Calculate the refilled state.
        SELECT * INTO refilled_state
        FROM _calculate_refilled_state(
            bucket_state.tickets,
            bucket_state.balance_as_of,
            temp_row.capacity,
            temp_row.refill_amount,
            temp_row.refill_interval
        );

        -- Store calculated values back into the temp table
        UPDATE temp_bucket_states
        SET refilled_tickets = refilled_state.available_tickets,
            new_balance_as_of = refilled_state.new_balance_as_of
        WHERE temp_bucket_states.key = temp_row.key;

        -- Check for failure.
        IF NOT failure_detected AND refilled_state.available_tickets < temp_row.requested THEN
            failure_detected := true;
        END IF;
    END LOOP;

    -- Pass 2: Commit changes based on whether a failure was detected.
    IF failure_detected THEN
        -- Failure: Update buckets to their refilled state without consuming tickets.
        FOR temp_row IN SELECT * FROM temp_bucket_states LOOP
            UPDATE resource_bucket
            SET tickets = temp_row.refilled_tickets,
                balance_as_of = temp_row.new_balance_as_of
            WHERE resource_bucket.key = temp_row.key;
        END LOOP;
    ELSE
        -- Success: Update all buckets with consumed amounts.
        FOR temp_row IN SELECT * FROM temp_bucket_states LOOP
            UPDATE resource_bucket
            SET tickets = temp_row.refilled_tickets - temp_row.requested,
                balance_as_of = temp_row.new_balance_as_of
            WHERE resource_bucket.key = temp_row.key;
        END LOOP;
    END IF;

    -- Pass 3: Return the final state for each key.
    RETURN QUERY
        SELECT
            t.key AS bucket_key,
            NOT failure_detected,
            CASE
                WHEN failure_detected THEN t.refilled_tickets
                ELSE t.refilled_tickets - t.requested
            END::BIGINT,
            CASE
                WHEN failure_detected THEN 0::BIGINT
                ELSE t.requested
            END::BIGINT
        FROM temp_bucket_states t;
END;
$$;

-- Function to atomically return tickets to multiple resource buckets.
CREATE OR REPLACE FUNCTION return_multiple_resource_tickets(
    p_keys TEXT[],
    p_amounts BIGINT[],
    p_capacities BIGINT[],
    p_refill_amounts BIGINT[],
    p_refill_intervals INTERVAL[]
)
RETURNS TABLE (bucket_key_returned TEXT, final_balance BIGINT)
LANGUAGE plpgsql AS $$
DECLARE
    i INT;
    bucket_state RECORD;
    refilled_state calculated_bucket_state;
    temp_row RECORD;
    new_balance BIGINT;
BEGIN
    -- Input validation: check for consistent array lengths
    IF array_length(p_keys, 1) IS NULL OR
       array_length(p_keys, 1) != array_length(p_amounts, 1) OR
       array_length(p_keys, 1) != array_length(p_capacities, 1) OR
       array_length(p_keys, 1) != array_length(p_refill_amounts, 1) OR
       array_length(p_keys, 1) != array_length(p_refill_intervals, 1) THEN
        RAISE EXCEPTION 'Input arrays must have the same length';
    END IF;

    -- Input validation: check for duplicate keys
    IF array_length(p_keys, 1) > 0 AND array_length(p_keys, 1) != (
        SELECT COUNT(DISTINCT key_column)
        FROM unnest(p_keys) AS keys(key_column)
    ) THEN
        RAISE EXCEPTION 'Duplicate keys are not allowed in the input array';
    END IF;

    -- Create a temporary table to store inputs
    CREATE TEMP TABLE temp_return_states (
        key TEXT PRIMARY KEY,
        amount BIGINT,
        capacity BIGINT,
        refill_amount BIGINT,
        refill_interval INTERVAL
    ) ON COMMIT DROP;

    -- Populate temp table from input arrays
    FOR i IN 1..array_length(p_keys, 1) LOOP
        INSERT INTO temp_return_states (key, amount, capacity, refill_amount, refill_interval)
        VALUES (p_keys[i], p_amounts[i], p_capacities[i], p_refill_amounts[i], p_refill_intervals[i]);
    END LOOP;

    -- Lock rows in a consistent order, calculate new state, and update.
    FOR temp_row IN SELECT * FROM temp_return_states ORDER BY temp_return_states.key LOOP
        -- Atomically get-or-create the bucket, locking the row.
        INSERT INTO resource_bucket (key, tickets, balance_as_of)
        VALUES (temp_row.key, temp_row.capacity, now())
        ON CONFLICT (key) DO UPDATE SET key = EXCLUDED.key
        RETURNING * INTO bucket_state;

        -- Calculate the refilled state.
        SELECT * INTO refilled_state
        FROM _calculate_refilled_state(
            bucket_state.tickets,
            bucket_state.balance_as_of,
            temp_row.capacity,
            temp_row.refill_amount,
            temp_row.refill_interval
        );

        -- Calculate the new balance, capped at capacity.
        new_balance := least(
            refilled_state.available_tickets + temp_row.amount,
            temp_row.capacity
        );

        -- Update the bucket with the new balance and timestamp.
        UPDATE resource_bucket
        SET
            tickets = new_balance,
            balance_as_of = refilled_state.new_balance_as_of
        WHERE resource_bucket.key = temp_row.key;

        -- Prepare the row for the return query.
        bucket_key_returned := temp_row.key;
        final_balance := new_balance;
        RETURN NEXT;
    END LOOP;
END;
$$;
