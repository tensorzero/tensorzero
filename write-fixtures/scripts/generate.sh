ulimit -n 99999
TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@127.0.0.1:8124/tensorzero cargo run --release -p write-fixtures -- --count 10000000 --config-file tensorzero-internal/tests/e2e/tensorzero.toml --function-name basic_test --variant-name random_answer --max-inflight 10 "My input prefix"

#TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@127.0.0.1:8124/tensorzero cargo run --profile release -p write-fixtures -- --count 10000000 --config-file tensorzero-internal/tests/e2e/tensorzero.toml --function-name basic_test --variant-name random_answer --max-inflight 1000 "My input prefix"


#perf record \
#    -e cycles \                             
#    -e sched:sched_switch --switch-events \
#    --sample-cpu \                          
#   -m 8M \                                
#    --aio -z \                             
#    --call-graph dwarf \                   
#    target/release-debug/write-fixtures --count 10000000 --config-file tensorzero-internal/tests/e2e/tensorzero.toml --function-name basic_test --variant-name random_answer --max-inflight 2000 'My input prefix'