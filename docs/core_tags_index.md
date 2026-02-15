# CORE Tag Index

This file lists all current `#CORE[...]` tags in the repository with exact file and line references.

## `v2ecore/emulator.py`

- `#CORE[C-THRESH-MISMATCH]` -> `v2ecore/emulator.py:502`
- `#CORE[G-THRESH-PROB-SCALE]` -> `v2ecore/emulator.py:520`
- `#CORE[F-LEAK-FPN]` -> `v2ecore/emulator.py:548`
- `#CORE[A-DELTA-T]` -> `v2ecore/emulator.py:705`
- `#CORE[D-LINLOG-CALL]` -> `v2ecore/emulator.py:716`
- `#CORE[E-INTEN-SCALE]` -> `v2ecore/emulator.py:726`
- `#CORE[E-LPF-CALL]` -> `v2ecore/emulator.py:741`
- `#CORE[G-PHOTO-NOISE-INJECT]` -> `v2ecore/emulator.py:749`
- `#CORE[F-LEAK-CALL]` -> `v2ecore/emulator.py:792`
- `#CORE[F-DIFF]` -> `v2ecore/emulator.py:809`
- `#CORE[F-EVENT-MAP-CALL]` -> `v2ecore/emulator.py:831`
- `#CORE[F-TS-SUBDIV]` -> `v2ecore/emulator.py:853`
- `#CORE[H-REFRACTORY-EXT]` -> `v2ecore/emulator.py:894`
- `#CORE[G-SHOT-CALL]` -> `v2ecore/emulator.py:956`
- `#CORE[F-LMEM-UPDATE]` -> `v2ecore/emulator.py:1002`
- `#CORE[G-SHOT-RESET]` -> `v2ecore/emulator.py:1009`

## `v2ecore/emulator_utils.py`

- `#CORE[D-LINLOG]` -> `v2ecore/emulator_utils.py:35`
- `#CORE[E-LPF-TAU]` -> `v2ecore/emulator_utils.py:82`
- `#CORE[E-LPF-EPS]` -> `v2ecore/emulator_utils.py:89`
- `#CORE[E-LPF-IIR]` -> `v2ecore/emulator_utils.py:106`
- `#CORE[F-LEAK-RATE]` -> `v2ecore/emulator_utils.py:133`
- `#CORE[F-LEAK-LMEM]` -> `v2ecore/emulator_utils.py:137`
- `#CORE[F-EVENT-QUANT]` -> `v2ecore/emulator_utils.py:164`
- `#CORE[G-PHOTO-VRMS-FIT]` -> `v2ecore/emulator_utils.py:223`
- `#CORE[G-SHOT-PROB]` -> `v2ecore/emulator_utils.py:340`
- `#CORE[G-SHOT-THRESH]` -> `v2ecore/emulator_utils.py:350`

## Quick Refresh Command

Use this to regenerate/check the raw list:

```bash
rg -n "#CORE\\[[^]]+\\]" v2ecore/emulator.py v2ecore/emulator_utils.py | sort
```
