#[cfg(test)]
mod test_optional_authentication {
    use std::sync::Arc;
    use tensorzero_internal::config_parser::{AuthenticationConfig, Config};
    use tensorzero_internal::gateway_util;

    #[tokio::test]
    async fn test_authentication_disabled() {
        // Create config with authentication disabled
        let mut config = Config::default();
        config.gateway.authentication = AuthenticationConfig {
            enabled: Some(false),
        };

        let app_state = gateway_util::AppStateData::new(Arc::new(config))
            .await
            .unwrap();

        // Verify authentication is disabled
        match &app_state.authentication_info {
            gateway_util::AuthenticationInfo::Disabled => {
                // Success - authentication is disabled as expected
            }
            gateway_util::AuthenticationInfo::Enabled(_) => {
                panic!("Expected authentication to be disabled");
            }
        }
    }

    #[tokio::test]
    async fn test_authentication_enabled_by_default() {
        // Create config with default authentication (not specified)
        let config = Config::default();

        let app_state = gateway_util::AppStateData::new(Arc::new(config))
            .await
            .unwrap();

        // Verify authentication is enabled by default
        match &app_state.authentication_info {
            gateway_util::AuthenticationInfo::Enabled(_) => {
                // Success - authentication is enabled by default
            }
            gateway_util::AuthenticationInfo::Disabled => {
                panic!("Expected authentication to be enabled by default");
            }
        }
    }

    #[tokio::test]
    async fn test_authentication_explicitly_enabled() {
        // Create config with authentication explicitly enabled
        let mut config = Config::default();
        config.gateway.authentication = AuthenticationConfig {
            enabled: Some(true),
        };

        let app_state = gateway_util::AppStateData::new(Arc::new(config))
            .await
            .unwrap();

        // Verify authentication is enabled
        match &app_state.authentication_info {
            gateway_util::AuthenticationInfo::Enabled(_) => {
                // Success - authentication is enabled as expected
            }
            gateway_util::AuthenticationInfo::Disabled => {
                panic!("Expected authentication to be enabled");
            }
        }
    }
}
