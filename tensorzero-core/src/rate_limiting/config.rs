use std::collections::HashMap;

struct RateLimitingConfig {
    rules: Vec<RateLimitingConfigRule>,
    enabled: bool, // default true, Postgres required if rules is nonempty.
}

struct RateLimitingConfigRule {
    limits: Vec<RateLimit>,
    scope: Vec<RateLimitingConfigScope>,
    priority: RateLimitingConfigPriority,
}

struct RateLimit {
    resource: RateLimitResource,
    interval: RateLimitInterval,
}

enum RateLimitingConfigPriority {
    Priority(usize),
    Always,
}

enum RateLimitResource {
    Token,
    Cent, // or something more granular?
}

enum RateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
} // implement a getter for a chrono::TimeDelta or something

enum RateLimitingConfigScope {
    Tag(TagRateLimitingConfigScope),
    // model_name = "my_model"
    // function_name = "my_function"
}

struct TagRateLimitingConfigScope {
    tag_key: String,
    tag_value: String,
}

impl RateLimitingConfigScope {
    fn matches(info: &ScopeInfo) -> bool {
        todo!()
    }
}

// utility struct to pass in at "check time"
struct ScopeInfo<'a> {
    function_name: &'a str,
    model_name: &'a str,
    tags: &'a HashMap<String, String>,
}

struct TagRateLimitScope {
    tag_key: String,
    tag_value: String,
}
