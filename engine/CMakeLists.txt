# Сборка движка через CMake (альтернатива Makefile): библиотека и утилита mv_cli.
cmake_minimum_required(VERSION 3.10)
project(matreshka_voice C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_library(matreshka_voice STATIC src/mv_features.c src/mv_nn.c src/matreshka_voice.c)
target_include_directories(matreshka_voice PUBLIC include)
# libm (математика) есть не везде отдельной библиотекой, поэтому подключаем, только если нашлась.
find_library(MATH_LIB m)
if(MATH_LIB)
  target_link_libraries(matreshka_voice PUBLIC ${MATH_LIB})
endif()
add_executable(mv_cli cli/mv_cli.c)
target_link_libraries(mv_cli matreshka_voice)
