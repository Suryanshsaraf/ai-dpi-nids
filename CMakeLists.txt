cmake_minimum_required(VERSION 3.16)
project(PacketAnalyzer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header files location
include_directories(${CMAKE_SOURCE_DIR}/include)

# Targets
# 1. Simple Packet Analyzer CLI
add_executable(packet_analyzer
    src/main.cpp
    src/pcap_reader.cpp
    src/packet_parser.cpp
)

# 2. Multi-threaded DPI Engine CLI
add_executable(dpi_engine
    src/main_dpi.cpp
    src/dpi_engine.cpp
    src/connection_tracker.cpp
    src/fast_path.cpp
    src/load_balancer.cpp
    src/packet_parser.cpp
    src/pcap_reader.cpp
    src/rule_manager.cpp
    src/sni_extractor.cpp
    src/types.cpp
)

# Thread library requirement for multi-threaded target
find_package(Threads REQUIRED)
target_link_libraries(dpi_engine PRIVATE Threads::Threads)
