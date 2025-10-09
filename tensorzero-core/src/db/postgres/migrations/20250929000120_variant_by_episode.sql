-- Create the variant_by_episode table
-- This table is used for advanced experimentation
CREATE TABLE variant_by_episode (
    function_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    variant_name TEXT NOT NULL,
    PRIMARY KEY (function_name, episode_id)
);
