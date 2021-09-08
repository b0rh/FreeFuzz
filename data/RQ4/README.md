# FreeFuzz RQ4

This directory contains the data for comparisons with the state-of-the-art work (LEMON and CRADLE).

We compare their sources of inputs and mutation strategies.

```
./show_comparison.sh
```

The output should be:

```
Input
  cradle_input: 28967 lines covered
  FreeFuzz_full_input: 33389 lines covered
  lemon_input: 29489 lines covered
  cradle_input: 59 APIs covered
  FreeFuzz_full_input: 313 APIs covered
  lemon_input: 30 APIs covered
Mutation
  FreeFuzz_full_mutation: 35473 lines covered
  FreeFuzz_model_only_mutation: 30488 lines covered
  lemon_mutation: 29766 lines covered
```
