# Copy to wire-os-victor/cmake/matreshka_voice.cmake and the engine to 3rd/matreshka_voice/
# (engine/include and engine/src from this repository).
set(MATRESHKA_VOICE_HOME "${CMAKE_SOURCE_DIR}/3rd/matreshka_voice")
if (NOT TARGET matreshka_voice)
  add_library(matreshka_voice STATIC
    "${MATRESHKA_VOICE_HOME}/src/mv_features.c"
    "${MATRESHKA_VOICE_HOME}/src/mv_nn.c"
    "${MATRESHKA_VOICE_HOME}/src/matreshka_voice.c")
  set_target_properties(matreshka_voice PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
  target_include_directories(matreshka_voice PUBLIC "${MATRESHKA_VOICE_HOME}/include")
  target_compile_options(matreshka_voice PRIVATE -O2 -ffast-math)
  target_link_libraries(matreshka_voice PUBLIC m)
endif()
