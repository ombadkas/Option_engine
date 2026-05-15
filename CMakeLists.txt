cmake_minimum_required(VERSION 3.20)
project(OptionsEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /WX- /O2 /fp:precise)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Core static library ───────────────────────────────────────────────────────
add_library(options_engine STATIC
    src/pricer.cpp
    src/mc_engine.cpp
    src/greeks.cpp
    src/vol_surface.cpp
    src/binomial_tree.cpp
)

target_include_directories(options_engine PUBLIC include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(options_engine PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(options_engine PUBLIC OMP_AVAILABLE)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Benchmarks (opt-in) ───────────────────────────────────────────────────────
option(BUILD_BENCHMARKS "Build Google Benchmark suite" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

# ── Python bindings (opt-in) ──────────────────────────────────────────────────
option(BUILD_PYTHON "Build pybind11 Python module" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(options_py python/bindings.cpp)
    target_link_libraries(options_py PRIVATE options_engine)
    target_compile_definitions(options_py PRIVATE VERSION_INFO="${PROJECT_VERSION}")
endif()
