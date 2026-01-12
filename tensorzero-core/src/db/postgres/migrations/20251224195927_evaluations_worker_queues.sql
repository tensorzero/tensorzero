-- Set up durable queues for evaluations_topk and autopilot
SELECT durable.create_queue('autopilot');
