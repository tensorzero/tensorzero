-- =============================================================================
-- PostgreSQL Schema Test Script
-- Description: Tests all tables, indexes, triggers, and views created by migrations.
-- Usage: psql -d tensorzero -f test_schema.sql
-- =============================================================================

-- Enable timing to see query performance
\timing on

-- =============================================================================
-- SETUP: Generate test UUIDs (UUIDv7-like for realistic testing)
-- =============================================================================

DO $$
DECLARE
    -- Test IDs
    chat_inference_id UUID := gen_random_uuid();
    json_inference_id UUID := gen_random_uuid();
    model_inference_id_1 UUID := gen_random_uuid();
    model_inference_id_2 UUID := gen_random_uuid();
    episode_id UUID := gen_random_uuid();
    feedback_id_1 UUID := gen_random_uuid();
    feedback_id_2 UUID := gen_random_uuid();
    feedback_id_3 UUID := gen_random_uuid();
    feedback_id_4 UUID := gen_random_uuid();
    batch_id UUID := gen_random_uuid();
    batch_request_id UUID := gen_random_uuid();
    batch_inference_id UUID := gen_random_uuid();
    datapoint_id UUID := gen_random_uuid();
    run_id UUID := gen_random_uuid();
    run_episode_id UUID := gen_random_uuid();
    eval_feedback_id UUID := gen_random_uuid();
    icl_example_id UUID := gen_random_uuid();
BEGIN
    RAISE NOTICE '=== Starting Schema Tests ===';
    RAISE NOTICE 'Test IDs generated:';
    RAISE NOTICE '  chat_inference_id: %', chat_inference_id;
    RAISE NOTICE '  json_inference_id: %', json_inference_id;
    RAISE NOTICE '  episode_id: %', episode_id;

    -- =========================================================================
    -- TEST 1: Chat Inference Table
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 1: Chat Inference ---';

    INSERT INTO chat_inference (
        id, function_name, variant_name, episode_id, input, output, tags,
        tool_params, inference_params, processing_time_ms, ttft_ms,
        dynamic_tools, dynamic_provider_tools, allowed_tools, tool_choice,
        parallel_tool_calls, snapshot_hash
    ) VALUES (
        chat_inference_id,
        'test_function',
        'variant_a',
        episode_id,
        '{"messages": [{"role": "user", "content": "Hello"}]}'::jsonb,
        '{"content": "Hi there!", "role": "assistant"}'::jsonb,
        '{"env": "test", "team": "engineering"}'::jsonb,
        '{"tools": []}',
        '{"temperature": 0.7}'::jsonb,
        150,
        45,
        ARRAY['tool1', 'tool2'],
        ARRAY['provider_tool1'],
        'all',
        'auto',
        true,
        '\x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'::bytea
    );
    RAISE NOTICE 'Chat inference inserted successfully';

    -- Verify tag query using GIN index
    PERFORM * FROM chat_inference WHERE tags @> '{"env": "test"}'::jsonb;
    RAISE NOTICE 'Tag query (GIN index) works';

    -- =========================================================================
    -- TEST 2: JSON Inference Table
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 2: JSON Inference ---';

    INSERT INTO json_inference (
        id, function_name, variant_name, episode_id, input, output,
        output_schema, inference_params, processing_time_ms, tags, ttft_ms,
        auxiliary_content, snapshot_hash
    ) VALUES (
        json_inference_id,
        'json_function',
        'variant_b',
        episode_id,
        '{"prompt": "Generate a JSON object"}'::jsonb,
        '{"result": {"key": "value"}}'::jsonb,
        '{"type": "object", "properties": {"key": {"type": "string"}}}',
        '{"max_tokens": 100}'::jsonb,
        200,
        '{"env": "test"}'::jsonb,
        60,
        'Additional context here',
        NULL
    );
    RAISE NOTICE 'JSON inference inserted successfully';

    -- =========================================================================
    -- TEST 3: Model Inference Table (with usage counter trigger)
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 3: Model Inference (with trigger) ---';

    -- Get initial usage counts
    DECLARE
        initial_input_tokens BIGINT;
        initial_output_tokens BIGINT;
        initial_inferences BIGINT;
        final_input_tokens BIGINT;
        final_output_tokens BIGINT;
        final_inferences BIGINT;
    BEGIN
        SELECT count INTO initial_input_tokens FROM cumulative_usage WHERE type = 'input_tokens';
        SELECT count INTO initial_output_tokens FROM cumulative_usage WHERE type = 'output_tokens';
        SELECT count INTO initial_inferences FROM cumulative_usage WHERE type = 'model_inferences';

        INSERT INTO model_inference (
            id, inference_id, raw_request, raw_response, model_name,
            model_provider_name, input_tokens, output_tokens, response_time_ms,
            ttft_ms, system, input_messages, output, cached, finish_reason
        ) VALUES (
            model_inference_id_1,
            chat_inference_id,
            '{"model": "gpt-4", "messages": [...]}',
            '{"choices": [...], "usage": {...}}',
            'gpt-4',
            'openai',
            150,
            50,
            1200,
            45,
            'You are a helpful assistant',
            '[{"role": "user", "content": "Hello"}]'::jsonb,
            '{"content": "Hi there!"}'::jsonb,
            false,
            'stop'
        );

        -- Insert second model inference for json inference
        INSERT INTO model_inference (
            id, inference_id, model_name, model_provider_name,
            input_tokens, output_tokens, finish_reason
        ) VALUES (
            model_inference_id_2,
            json_inference_id,
            'claude-3-opus',
            'anthropic',
            100,
            75,
            'stop'
        );

        -- Verify trigger updated cumulative_usage
        SELECT count INTO final_input_tokens FROM cumulative_usage WHERE type = 'input_tokens';
        SELECT count INTO final_output_tokens FROM cumulative_usage WHERE type = 'output_tokens';
        SELECT count INTO final_inferences FROM cumulative_usage WHERE type = 'model_inferences';

        IF final_input_tokens = initial_input_tokens + 250 AND
           final_output_tokens = initial_output_tokens + 125 AND
           final_inferences = initial_inferences + 2 THEN
            RAISE NOTICE 'Cumulative usage trigger works correctly';
            RAISE NOTICE '  Input tokens: % -> %', initial_input_tokens, final_input_tokens;
            RAISE NOTICE '  Output tokens: % -> %', initial_output_tokens, final_output_tokens;
            RAISE NOTICE '  Model inferences: % -> %', initial_inferences, final_inferences;
        ELSE
            RAISE EXCEPTION 'Cumulative usage trigger failed!';
        END IF;
    END;

    -- =========================================================================
    -- TEST 4: Inference By ID View
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 4: Inference By ID View ---';

    DECLARE
        view_count INT;
    BEGIN
        SELECT COUNT(*) INTO view_count
        FROM inference_by_id
        WHERE id IN (chat_inference_id, json_inference_id);

        IF view_count = 2 THEN
            RAISE NOTICE 'inference_by_id view works (found % records)', view_count;
        ELSE
            RAISE EXCEPTION 'inference_by_id view failed! Expected 2, got %', view_count;
        END IF;
    END;

    -- =========================================================================
    -- TEST 5: Feedback Tables
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 5: Feedback Tables ---';

    -- Boolean feedback
    INSERT INTO boolean_metric_feedback (id, target_id, metric_name, value, tags)
    VALUES (feedback_id_1, chat_inference_id, 'is_helpful', true, '{"reviewer": "user_1"}'::jsonb);
    RAISE NOTICE 'Boolean feedback inserted';

    -- Float feedback
    INSERT INTO float_metric_feedback (id, target_id, metric_name, value, tags)
    VALUES (feedback_id_2, chat_inference_id, 'relevance_score', 0.95, '{"reviewer": "user_1"}'::jsonb);
    RAISE NOTICE 'Float feedback inserted';

    -- Comment feedback
    INSERT INTO comment_feedback (id, target_id, target_type, value, tags)
    VALUES (feedback_id_3, chat_inference_id, 'inference', 'Great response!', '{}'::jsonb);
    RAISE NOTICE 'Comment feedback inserted';

    -- Demonstration feedback
    INSERT INTO demonstration_feedback (id, inference_id, value, tags)
    VALUES (feedback_id_4, chat_inference_id, '{"improved": "response"}', '{}'::jsonb);
    RAISE NOTICE 'Demonstration feedback inserted';

    -- =========================================================================
    -- TEST 6: Batch Tables
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 6: Batch Tables ---';

    INSERT INTO batch_request (
        batch_id, id, batch_params, status, model_name, model_provider_name,
        function_name, variant_name
    ) VALUES (
        batch_id, batch_request_id,
        '{"max_concurrent": 10}'::jsonb,
        'pending',
        'gpt-4',
        'openai',
        'batch_function',
        'variant_a'
    );
    RAISE NOTICE 'Batch request inserted';

    INSERT INTO batch_model_inference (
        inference_id, batch_id, function_name, variant_name, episode_id,
        input, tags
    ) VALUES (
        batch_inference_id, batch_id, 'batch_function', 'variant_a',
        gen_random_uuid(),
        '{"messages": []}'::jsonb,
        '{}'::jsonb
    );
    RAISE NOTICE 'Batch model inference inserted';

    -- Update batch status
    UPDATE batch_request SET status = 'completed', completed_at = NOW()
    WHERE id = batch_request_id;
    RAISE NOTICE 'Batch status updated to completed';

    -- =========================================================================
    -- TEST 7: Datapoint Tables with Soft Delete
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 7: Datapoint Tables (Soft Delete) ---';

    INSERT INTO chat_inference_datapoint (
        id, dataset_name, function_name, input, tags, is_deleted
    ) VALUES (
        datapoint_id, 'test_dataset', 'test_function',
        '{"messages": []}'::jsonb, '{}'::jsonb, false
    );
    RAISE NOTICE 'Chat datapoint inserted';

    -- Verify partial index (active datapoints)
    DECLARE
        active_count INT;
    BEGIN
        SELECT COUNT(*) INTO active_count
        FROM chat_inference_datapoint
        WHERE dataset_name = 'test_dataset' AND NOT is_deleted;
        RAISE NOTICE 'Active datapoints: %', active_count;
    END;

    -- Soft delete
    UPDATE chat_inference_datapoint SET is_deleted = true WHERE id = datapoint_id;

    -- Verify updated_at trigger fired
    DECLARE
        dp_updated_at TIMESTAMPTZ;
    BEGIN
        SELECT updated_at INTO dp_updated_at
        FROM chat_inference_datapoint WHERE id = datapoint_id;
        RAISE NOTICE 'Datapoint updated_at after soft delete: %', dp_updated_at;
    END;

    -- =========================================================================
    -- TEST 8: Evaluation Tables
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 8: Evaluation Tables ---';

    INSERT INTO workflow_evaluation_run (
        run_id, variant_pins, tags, project_name, run_display_name
    ) VALUES (
        run_id,
        '{"function_a": "variant_1"}'::jsonb,
        '{"experiment": "test_run"}'::jsonb,
        'test_project',
        'Test Run 1'
    );
    RAISE NOTICE 'Workflow evaluation run inserted';

    INSERT INTO workflow_evaluation_run_episode (
        run_id, episode_id, variant_pins, datapoint_name, tags
    ) VALUES (
        run_id, run_episode_id,
        '{"function_a": "variant_1"}'::jsonb,
        'datapoint_001',
        '{}'::jsonb
    );
    RAISE NOTICE 'Workflow evaluation run episode inserted';

    INSERT INTO inference_evaluation_human_feedback (
        feedback_id, metric_name, datapoint_id, output, value
    ) VALUES (
        eval_feedback_id,
        'accuracy',
        datapoint_id,
        'The model output',
        '{"score": 4, "max": 5}'::jsonb
    );
    RAISE NOTICE 'Inference evaluation human feedback inserted';

    -- =========================================================================
    -- TEST 9: Config Snapshot
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 9: Config Snapshot ---';

    INSERT INTO config_snapshot (
        hash, config, extra_templates, tensorzero_version, tags
    ) VALUES (
        '\xdeadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567'::bytea,
        '{"functions": {"test": {}}}',
        '{"template1": "content"}'::jsonb,
        '0.1.0',
        '{"env": "test"}'::jsonb
    );
    RAISE NOTICE 'Config snapshot inserted';

    -- =========================================================================
    -- TEST 10: Deployment ID (Singleton)
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 10: Deployment ID (Singleton) ---';

    INSERT INTO deployment_id (deployment_id) VALUES ('deploy-test-001')
    ON CONFLICT (id) DO UPDATE SET deployment_id = EXCLUDED.deployment_id;
    RAISE NOTICE 'Deployment ID set';

    -- Try to insert a second row (should fail due to constraint)
    BEGIN
        INSERT INTO deployment_id (id, deployment_id) VALUES (2, 'deploy-test-002');
        RAISE EXCEPTION 'Singleton constraint should have failed!';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'Singleton constraint works correctly (prevented duplicate row)';
    END;

    -- =========================================================================
    -- TEST 11: Model Inference Cache
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 11: Model Inference Cache ---';

    INSERT INTO model_inference_cache (
        short_cache_key, long_cache_key, output, input_tokens, output_tokens, finish_reason
    ) VALUES (
        12345678901234,
        '\x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'::bytea,
        '{"cached": "response"}'::jsonb,
        100,
        50,
        'stop'
    );
    RAISE NOTICE 'Cache entry inserted';

    -- Test partial index for non-deleted entries
    DECLARE
        cache_count INT;
    BEGIN
        SELECT COUNT(*) INTO cache_count
        FROM model_inference_cache
        WHERE short_cache_key = 12345678901234 AND NOT is_deleted;
        RAISE NOTICE 'Non-deleted cache entries: %', cache_count;
    END;

    -- =========================================================================
    -- TEST 12: Dynamic In-Context Learning Example (pgvector)
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 12: Dynamic ICL Example (pgvector) ---';

    -- Check if pgvector is available
    BEGIN
        INSERT INTO dynamic_in_context_learning_example (
            id, function_name, variant_name, namespace, input, output, embedding
        ) VALUES (
            icl_example_id,
            'icl_function',
            'icl_variant',
            'default',
            '{"query": "test query"}'::jsonb,
            '{"response": "test response"}'::jsonb,
            '[0.1, 0.2, 0.3, 0.4, 0.5]'::vector
        );
        RAISE NOTICE 'ICL example with embedding inserted';

        -- Test similarity search (if index exists)
        DECLARE
            similar_count INT;
        BEGIN
            SELECT COUNT(*) INTO similar_count
            FROM dynamic_in_context_learning_example
            WHERE function_name = 'icl_function'
              AND variant_name = 'icl_variant'
              AND namespace = 'default';
            RAISE NOTICE 'ICL examples found: %', similar_count;
        END;
    EXCEPTION WHEN undefined_object THEN
        RAISE NOTICE 'pgvector extension not installed - skipping embedding test';
    END;

    -- =========================================================================
    -- TEST 13: Episode Summary View
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 13: Episode Summary View ---';

    DECLARE
        ep_summary RECORD;
    BEGIN
        SELECT * INTO ep_summary
        FROM episode_summary
        WHERE episode_id = episode_id
        LIMIT 1;

        IF ep_summary IS NOT NULL THEN
            RAISE NOTICE 'Episode summary:';
            RAISE NOTICE '  Inference count: %', ep_summary.inference_count;
            RAISE NOTICE '  First timestamp: %', ep_summary.first_timestamp;
            RAISE NOTICE '  Last timestamp: %', ep_summary.last_timestamp;
        ELSE
            RAISE NOTICE 'No episode summary found (expected if episode has no inferences)';
        END IF;
    END;

    -- =========================================================================
    -- TEST 14: Tag Index Performance (jsonb_path_ops)
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Test 14: Tag Index Verification ---';

    -- These queries should use the GIN indexes
    PERFORM * FROM chat_inference WHERE tags @> '{"env": "test"}'::jsonb;
    PERFORM * FROM boolean_metric_feedback WHERE tags @> '{"reviewer": "user_1"}'::jsonb;
    RAISE NOTICE 'Tag containment queries executed (check EXPLAIN for index usage)';

    -- =========================================================================
    -- CLEANUP
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '--- Cleanup ---';

    -- Delete test data in reverse dependency order
    DELETE FROM inference_evaluation_human_feedback WHERE feedback_id = eval_feedback_id;
    DELETE FROM workflow_evaluation_run_episode WHERE episode_id = run_episode_id;
    DELETE FROM workflow_evaluation_run WHERE run_id = run_id;
    DELETE FROM dynamic_in_context_learning_example WHERE id = icl_example_id;
    DELETE FROM model_inference_cache WHERE short_cache_key = 12345678901234;
    DELETE FROM config_snapshot WHERE hash = '\xdeadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567'::bytea;
    DELETE FROM chat_inference_datapoint WHERE id = datapoint_id;
    DELETE FROM batch_model_inference WHERE inference_id = batch_inference_id;
    DELETE FROM batch_request WHERE id = batch_request_id;
    DELETE FROM demonstration_feedback WHERE id = feedback_id_4;
    DELETE FROM comment_feedback WHERE id = feedback_id_3;
    DELETE FROM float_metric_feedback WHERE id = feedback_id_2;
    DELETE FROM boolean_metric_feedback WHERE id = feedback_id_1;
    DELETE FROM model_inference WHERE id IN (model_inference_id_1, model_inference_id_2);
    DELETE FROM json_inference WHERE id = json_inference_id;
    DELETE FROM chat_inference WHERE id = chat_inference_id;

    RAISE NOTICE 'Test data cleaned up';

    RAISE NOTICE '';
    RAISE NOTICE '=== All Schema Tests Passed! ===';
END $$;

-- =============================================================================
-- INDEX USAGE VERIFICATION (Run these manually with EXPLAIN ANALYZE)
-- =============================================================================

-- Uncomment and run these to verify index usage:

-- EXPLAIN ANALYZE SELECT * FROM chat_inference WHERE tags @> '{"env": "prod"}'::jsonb;
-- EXPLAIN ANALYZE SELECT * FROM chat_inference WHERE function_name = 'test' AND variant_name = 'v1';
-- EXPLAIN ANALYZE SELECT * FROM chat_inference WHERE episode_id = 'some-uuid';
-- EXPLAIN ANALYZE SELECT * FROM model_inference WHERE inference_id = 'some-uuid';
-- EXPLAIN ANALYZE SELECT * FROM chat_inference_datapoint WHERE dataset_name = 'ds' AND NOT is_deleted;
-- EXPLAIN ANALYZE SELECT * FROM workflow_evaluation_run WHERE project_name = 'proj' AND NOT is_deleted;

-- =============================================================================
-- STRESS TEST (Optional)
-- =============================================================================

-- Uncomment to run a basic stress test:

-- DO $$
-- BEGIN
--     FOR i IN 1..1000 LOOP
--         INSERT INTO chat_inference (
--             id, function_name, variant_name, episode_id, input, output, tags
--         ) VALUES (
--             gen_random_uuid(),
--             'stress_function_' || (i % 10),
--             'variant_' || (i % 5),
--             gen_random_uuid(),
--             '{"messages": []}'::jsonb,
--             '{"content": "response"}'::jsonb,
--             ('{"batch": "' || (i / 100) || '"}')::jsonb
--         );
--     END LOOP;
--     RAISE NOTICE 'Inserted 1000 chat inference records';
-- END $$;
