#!/bin/sh
# Dynamic entrypoint based on TRIOS_TRAINER_BIN variable
BIN="${TRIOS_TRAINER_BIN:-entrypoint}"

case "$BIN" in
    scarab)
        exec scarab
        ;;
    trios-train)
        exec trios-train "$@"
        ;;
    *)
        exec entrypoint "$@"
        ;;
esac
