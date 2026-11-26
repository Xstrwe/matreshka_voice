# Скопируйте этот файл в wire-os-victor/cmake/matreshka_voice.cmake, а движок — в 3rd/matreshka_voice/
# (папки engine/include и engine/src из этого репозитория).
# Файл описывает статическую библиотеку matreshka_voice из трёх .c файлов движка.
# if (NOT TARGET ...) защищает от повторного объявления, если файл подключат дважды.
set(MATRESHKA_VOICE_HOME "${CMAKE_SOURCE_DIR}/3rd/matreshka_voice")
if (NOT TARGET matreshka_voice)
  add_library(matreshka_voice STATIC
    "${MATRESHKA_VOICE_HOME}/src/mv_features.c"
    "${MATRESHKA_VOICE_HOME}/src/mv_nn.c"
    "${MATRESHKA_VOICE_HOME}/src/matreshka_voice.c")
  set_target_properties(matreshka_voice PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
  target_include_directories(matreshka_voice PUBLIC "${MATRESHKA_VOICE_HOME}/include")
  # -ffast-math разрешает компилятору переставлять операции с float ради скорости;
  # результат отличается от эталона лишь в последних знаках.
  target_compile_options(matreshka_voice PRIVATE -O2 -ffast-math)
  target_link_libraries(matreshka_voice PUBLIC m)   # libm — математические функции (expf, logf)
endif()
