# Lumberjack compiler

This repository contains the random forest model compiler for the Lumberjack RF hardware accelerator.

```
Usage: lumberjack-compiler <COMMAND>

Commands:
  build               Compile a .ljmodel from a CSV definition
  build-cache-data    Compile a Lumberjack model from a CSV definition, splitting it into per-cell cache .ljcache files
  build-test-vectors  Compile .ljdata test vectors from a CSV model file, and a CSV test vectors file
  analyze             Analyze a compiled .ljmodel model
  help                Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```
