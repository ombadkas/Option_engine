# Requires Google Benchmark: https://github.com/google/benchmark
# cmake -DBUILD_BENCHMARKS=ON -Dbenchmark_DIR=<path> ..
find_package(benchmark REQUIRED)

add_executable(latency_bench latency_bench.cpp)
target_link_libraries(latency_bench PRIVATE options_engine benchmark::benchmark)
