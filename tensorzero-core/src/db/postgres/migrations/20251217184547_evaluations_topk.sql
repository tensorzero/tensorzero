-- Create the evaluations_topk queue for durable workflow execution
SELECT durable.create_queue('evaluations_topk');
