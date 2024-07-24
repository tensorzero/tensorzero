use api::clickhouse::ClickHouseConnectionInfo;

pub async fn clickhouse_flush_async_insert(connection_info: &ClickHouseConnectionInfo) {
    let (url, client) = match connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unimplemented!(),
        ClickHouseConnectionInfo::Production { url, client } => (url.clone(), client),
    };
    client
        .post(url)
        .body("SYSTEM FLUSH ASYNC INSERT QUEUE")
        .send()
        .await
        .expect("Failed to flush ClickHouse");
}
