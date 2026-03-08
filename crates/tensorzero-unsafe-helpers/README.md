A helper crate to hold 'unsafe' code (which we normally forbid)
This should only be used by tests, not in production code

Currently, this holds safe wrappers for 'tensorzero_unsafe_helpers::set_env_var_tests_only' and 'tensorzero_unsafe_helpers::remove_env_var_tests_only'
