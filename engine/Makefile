# Движок matreshka_voice. Чистый C99, зависит только от libc и libm.
#   make                 сборка для этой машины (библиотека + mv_cli)
#   make test            проверка совпадения с Python и смоук-тесты (нужно окружение Python, см. ../README.md)
#   make armv7           кросс-сборка для Vector (Cortex-A7, NEON), нужен gcc или clang для arm-linux-gnueabi(hf)
# ?= задаёт значение, только если переменную не передали снаружи (например, CC=clang make).
# override += добавляет наши флаги, даже если CFLAGS указали в командной строке.
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2
override CFLAGS += -std=c99 -Wall -Wextra -pedantic -Iinclude
LDLIBS  += -lm

SRC = src/mv_features.c src/mv_nn.c src/matreshka_voice.c
OBJ = $(SRC:.c=.o)
LIB = libmatreshka_voice.a

all: $(LIB) mv_cli

# Статическая библиотека — архив объектных файлов. В правилах: $@ — цель,
# $^ — все зависимости, $< — первая зависимость.
$(LIB): $(OBJ)
	$(AR) rcs $@ $^

# Любой .c пересобирается и при изменении заголовков.
src/%.o: src/%.c include/matreshka_voice.h src/mv_internal.h
	$(CC) $(CFLAGS) -c $< -o $@

mv_cli: cli/mv_cli.c $(LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $< $(LIB) $(LDLIBS) -o $@

# В Vector стоит APQ8009 — четыре ядра Cortex-A7. Программы WireOS передают float
# через целые регистры (soft-float ABI), но считают на аппаратном FPU/NEON
# (armv7a-linux-gnueabi, -mfloat-abi=softfp). Собираем статически, чтобы не зависеть
# от библиотек на роботе.
ARM_CC     ?= arm-linux-gnueabi-gcc
ARM_CFLAGS ?= -O2 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=softfp -ffast-math
armv7:
	$(MAKE) clean-obj
	$(MAKE) CC="$(ARM_CC)" AR="$(ARM_AR)" CFLAGS="$(ARM_CFLAGS)" LDFLAGS="-static" all
	mkdir -p arm && mv $(LIB) mv_cli arm/
	$(MAKE) clean-obj
ARM_AR ?= $(subst gcc,ar,$(ARM_CC))

test: all
	sh tests/run_tests.sh

clean-obj:
	rm -f $(OBJ)

clean: clean-obj
	rm -f $(LIB) mv_cli
	rm -rf arm

.PHONY: all test armv7 clean clean-obj
