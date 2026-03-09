use reqwest::Url;

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("TENSORZERO_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3000".to_string());
}

pub fn get_gateway_endpoint(endpoint: &str) -> Url {
    let base_url: Url = GATEWAY_URL
        .parse()
        .expect("Invalid gateway URL (check environment variable TENSORZERO_GATEWAY_URL)");

    base_url.join(endpoint).unwrap()
}
