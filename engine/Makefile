# matreshka_voice engine. Plain C99, depends only on libc and libm.
#   make                 host build (library + mv_cli)
#   make test            parity and smoke tests (needs the Python venv, see ../README.md)
#   make armv7           cross build for Vector (Cortex-A7, NEON), needs an arm-linux-gnueabi(hf) gcc or clang
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2
override CFLAGS += -std=c99 -Wall -Wextra -pedantic -Iinclude
LDLIBS  += -lm

SRC = src/mv_features.c src/mv_nn.c src/matreshka_voice.c
OBJ = $(SRC:.c=.o)
LIB = libmatreshka_voice.a

all: $(LIB) mv_cli

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

src/%.o: src/%.c include/matreshka_voice.h src/mv_internal.h
	$(CC) $(CFLAGS) -c $< -o $@

mv_cli: cli/mv_cli.c $(LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $< $(LIB) $(LDLIBS) -o $@

# Vector's APQ8009 is a quad Cortex-A7. WireOS userland uses the soft-float calling
# convention with hardware FP/NEON (armv7a-linux-gnueabi, -mfloat-abi=softfp).
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
