#!lua name=tensorzero_ratelimit_v2

-- IMPORTANT: If you change the implementation of any function in a way that affects
-- its behavior or return format, update the function name version suffix (e.g., v1 -> v2)
-- to avoid rolling deploy issues where old Gateway instances call new function code.
--
-- Additionally, the response types for these functions must match what Gateway expects
-- in tensorzero-core/src/db/valkey/rate_limiting.rs.
--
-- Lua doesn't have real "array" types; its arrays are associative arrays and by convention
-- use 1-based indexing. See https://github.com/valkey-io/valkey-doc/blob/main/topics/functions-intro.md
--
-- Rate limiting state in Valkey is stored as a Hash:
-- key = 'tensorzero_ratelimit:<key>',
-- value = { 'balance': balance, 'last_refilled': timestamp at microseconds precision }

-- Lint.IfEdited()

-- Minimum TTL in seconds to avoid very short expirations
local MIN_TTL_SECONDS = 3600  -- 1 hour

-- Rate limiting key prefixes
local RATE_LIMITING_KEY_PREFIX = 'tensorzero_ratelimit:'
local OLD_RATE_LIMITING_KEY_PREFIX_FOR_MIGRATION = 'ratelimit:'

-- Get current server time in microseconds
local function get_server_time_micros()
    local time = server.call('TIME')
    return tonumber(time[1]) * 1000000 + tonumber(time[2])
end

-- Calculate TTL for a rate limit key based on refill parameters.
-- TTL = 2 * (time to fully refill from zero), with a minimum of MIN_TTL_SECONDS.
-- This ensures inactive keys expire but active keys are refreshed on each write.
local function calculate_ttl_seconds(capacity, refill_amount, refill_interval_micros)
    if refill_amount <= 0 or refill_interval_micros <= 0 then
        return MIN_TTL_SECONDS
    end
    local intervals_to_fill = math.ceil(capacity / refill_amount)
    local time_to_fill_micros = intervals_to_fill * refill_interval_micros
    local time_to_fill_seconds = math.ceil(time_to_fill_micros / 1000000)
    return math.max(MIN_TTL_SECONDS, 2 * time_to_fill_seconds)
end

-- Shared helper: apply token refill based on elapsed time
-- Returns: new_balance, new_last_refilled
-- Note: new_last_refilled is advanced by whole intervals only, keeping refills aligned
local function apply_refill(balance, last_refilled, now, capacity, refill_amount, refill_interval)
    if refill_amount <= 0 or refill_interval <= 0 then
        return balance, last_refilled
    end
    local elapsed = now - last_refilled
    -- Handle backward clock steps by clamping elapsed to 0
    if elapsed < 0 then
        elapsed = 0
    end
    local intervals = math.floor(elapsed / refill_interval)
    local new_balance = math.min(balance + intervals * refill_amount, capacity)
    local new_last_refilled = last_refilled + intervals * refill_interval
    return new_balance, new_last_refilled
end

-- Function 1: consume_tickets
-- Atomically consumes tokens from multiple buckets with all-or-nothing semantics.
-- KEYS: List of rate limit keys
-- ARGV: Flattened [requested, capacity, refill_amount, refill_interval_micros] per key
-- Returns: JSON array of {key, success, remaining, consumed}
local function consume_tickets(keys, args)
    local num_keys = #keys
    local now = get_server_time_micros()

    -- Input validation: check for duplicate keys
    local seen_keys = {}
    for i = 1, num_keys do
        local key = keys[i]
        if seen_keys[key] then
            error('Duplicate keys are not allowed in the input array')
        end
        seen_keys[key] = true
    end

    -- Phase 1: Read all keys and calculate refilled balances
    local state = {}  -- {balance, last_refilled} per key after refill
    local params = {}

    for i = 1, num_keys do
        local key = keys[i]
        local base_idx = (i - 1) * 4
        local requested = tonumber(args[base_idx + 1])
        local capacity = tonumber(args[base_idx + 2])
        local refill_amount = tonumber(args[base_idx + 3])
        local refill_interval = tonumber(args[base_idx + 4])

        params[i] = {
            requested = requested,
            capacity = capacity,
            refill_amount = refill_amount,
            refill_interval = refill_interval
        }

        -- HMGET reads multiple fields from the hash in one call, and returns a table with two values:
        -- [balance, last_refilled]
        -- Returns [nil, nil] if the key does not exist.
        local data = server.call('HMGET', RATE_LIMITING_KEY_PREFIX .. key, 'balance', 'last_refilled')
        local balance = tonumber(data[1]) or capacity  -- New bucket starts at capacity
        local last_refilled = tonumber(data[2]) or now

        local new_balance, new_last_refilled = apply_refill(
            balance, last_refilled, now, capacity, refill_amount, refill_interval)
        state[i] = { balance = new_balance, last_refilled = new_last_refilled }
    end

    -- Phase 2: Check if ALL requests can be satisfied
    local can_satisfy_all = true
    for i = 1, num_keys do
        if params[i].requested > state[i].balance then
            can_satisfy_all = false
            break
        end
    end

    -- Phase 3: Build results
    local results = {}

    if can_satisfy_all then
        -- All succeed: deduct from all and persist
        for i = 1, num_keys do
            local key = keys[i]
            local redis_key = RATE_LIMITING_KEY_PREFIX .. key
            local new_balance = state[i].balance - params[i].requested

            -- Store aligned last_refilled, NOT current time
            server.call('HSET', redis_key,
                'balance', new_balance,
                'last_refilled', state[i].last_refilled)

            -- Set TTL to expire inactive keys
            local ttl = calculate_ttl_seconds(
                params[i].capacity, params[i].refill_amount, params[i].refill_interval)
            server.call('EXPIRE', redis_key, ttl)

            results[i] = {
                key = key,
                success = true,
                remaining = new_balance,
                consumed = params[i].requested
            }
        end
    else
        -- All fail: return current balances, no writes
        for i = 1, num_keys do
            results[i] = {
                key = keys[i],
                success = false,
                remaining = state[i].balance,
                consumed = 0
            }
        end
    end

    return cjson.encode(results)
end

-- Function 2: return_tickets
-- Returns tokens to buckets, capped at capacity.
-- KEYS: List of rate limit keys
-- ARGV: Flattened [returned, capacity, refill_amount, refill_interval_micros] per key
-- Returns: JSON array of {key, balance}
local function return_tickets(keys, args)
    local num_keys = #keys
    local now = get_server_time_micros()
    local results = {}

    -- Input validation: check for duplicate keys
    local seen_keys = {}
    for i = 1, num_keys do
        local key = keys[i]
        if seen_keys[key] then
            error('Duplicate keys are not allowed in the input array')
        end
        seen_keys[key] = true
    end

    for i = 1, num_keys do
        local key = keys[i]
        local redis_key = RATE_LIMITING_KEY_PREFIX .. key
        local base_idx = (i - 1) * 4
        local returned = tonumber(args[base_idx + 1])
        local capacity = tonumber(args[base_idx + 2])
        local refill_amount = tonumber(args[base_idx + 3])
        local refill_interval = tonumber(args[base_idx + 4])

        local data = server.call('HMGET', redis_key, 'balance', 'last_refilled')
        local balance = tonumber(data[1]) or capacity
        local last_refilled = tonumber(data[2]) or now

        local new_balance, new_last_refilled = apply_refill(
            balance, last_refilled, now, capacity, refill_amount, refill_interval)

        -- Add returned tokens, capped at capacity
        new_balance = math.min(new_balance + returned, capacity)

        -- Store aligned last_refilled, NOT current time
        server.call('HSET', redis_key,
            'balance', new_balance,
            'last_refilled', new_last_refilled)

        -- Set TTL to expire inactive keys
        local ttl = calculate_ttl_seconds(capacity, refill_amount, refill_interval)
        server.call('EXPIRE', redis_key, ttl)

        results[i] = {
            key = key,
            balance = new_balance
        }
    end

    return cjson.encode(results)
end

-- Function 3: get_balance
-- Read-only balance query with refill calculation (does NOT persist changes).
-- KEYS[1]: rate limit key
-- ARGV: [capacity, refill_amount, refill_interval_micros]
-- Returns: JSON object {balance}
local function get_balance(keys, args)
    local key = keys[1]
    local capacity = tonumber(args[1])
    local refill_amount = tonumber(args[2])
    local refill_interval = tonumber(args[3])
    local now = get_server_time_micros()

    local data = server.call('HMGET', RATE_LIMITING_KEY_PREFIX .. key, 'balance', 'last_refilled')
    local balance = tonumber(data[1]) or capacity
    local last_refilled = tonumber(data[2]) or now

    local new_balance, _ = apply_refill(
        balance, last_refilled, now, capacity, refill_amount, refill_interval)
    return cjson.encode({ balance = new_balance })
end

-- Function 4: migrate_old_keys
-- Migrates old rate limit keys ('ratelimit:*') to new prefixed keys ('tensorzero_ratelimit:*').
-- This preserves existing rate limit state during upgrades from older versions.
-- Keys are only copied if the new key doesn't already exist.
-- KEYS: (none)
-- ARGV: (none)
-- Returns: JSON object {migrated_count}
local function migrate_old_keys(keys, args)
    local migrated_count = 0
    local cursor = '0'

    repeat
        -- SCAN for old keys
        local result = server.call('SCAN', cursor, 'MATCH', OLD_RATE_LIMITING_KEY_PREFIX_FOR_MIGRATION .. '*', 'COUNT', 100)
        cursor = result[1]
        local found_keys = result[2]

        for _, old_key in ipairs(found_keys) do
            -- Extract the suffix after the old prefix
            local suffix = string.sub(old_key, #OLD_RATE_LIMITING_KEY_PREFIX_FOR_MIGRATION + 1)
            local new_key = RATE_LIMITING_KEY_PREFIX .. suffix

            -- Only migrate if new key doesn't exist
            local exists = server.call('EXISTS', new_key)
            if exists == 0 then
                -- Copy hash data
                local data = server.call('HGETALL', old_key)
                if #data > 0 then
                    server.call('HSET', new_key, unpack(data))

                    -- Copy TTL if present
                    local ttl = server.call('TTL', old_key)
                    if ttl > 0 then
                        server.call('EXPIRE', new_key, ttl)
                    end

                    migrated_count = migrated_count + 1
                end
            end
        end
    until cursor == '0'

    return cjson.encode({ migrated_count = migrated_count })
end

-- Register all functions with version suffix for safe rolling deploys
server.register_function('tensorzero_consume_tickets_v2', consume_tickets)
server.register_function('tensorzero_return_tickets_v2', return_tickets)
server.register_function{
    function_name = 'tensorzero_get_balance_v2',
    callback = get_balance,
    flags = { 'no-writes' }  -- Enables FCALL_RO on read replicas
}
server.register_function('tensorzero_migrate_old_keys_v1', migrate_old_keys)

-- Lint.ThenEdit(tensorzero-core/src/db/valkey/rate_limiting.rs)
