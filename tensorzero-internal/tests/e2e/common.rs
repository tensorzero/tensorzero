#![allow(clippy::print_stdout)]

use reqwest::Url;

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("GATEWAY_URL").unwrap_or("http://localhost:3000".to_string());
}

pub fn get_gateway_endpoint(endpoint: &str) -> Url {
    let base_url: Url = GATEWAY_URL
        .parse()
        .expect("Invalid gateway URL (check environment variable GATEWAY_URL)");

    base_url.join(endpoint).unwrap()
}
