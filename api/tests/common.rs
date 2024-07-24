use api::clickhouse::ClickHouseConnectionInfo;

pub async fn clickhouse_flush_async_insert(
    client: &reqwest::Client,
    connection_info: &ClickHouseConnectionInfo,
) {
    let url = match connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unimplemented!(),
        ClickHouseConnectionInfo::Production { url } => url.clone(),
    };
    client
        .post(url)
        .body("SYSTEM FLUSH ASYNC INSERT QUEUE")
        .send()
        .await
        .expect("Failed to flush ClickHouse");
}
