add_executable(test_pricer test_pricer.cpp)
target_link_libraries(test_pricer PRIVATE options_engine)

add_executable(test_greeks test_greeks.cpp)
target_link_libraries(test_greeks PRIVATE options_engine)

add_test(NAME PricerTests  COMMAND test_pricer)
add_test(NAME GreeksTests  COMMAND test_greeks)
