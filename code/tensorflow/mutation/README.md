# Mutation

We provide an example script to apply our fuzzing algorithm on `tf.keras.layers.Conv2D`:

```
python -m mutation.demo
```

The output code snippets are stored in `output/`.

 - `example_Conv2D.py`: the python file containing an invocation of `tf.keras.layers.Conv2D`
 - `example_Conv2D_mutation.py`: the python file containing a mutated invocation of `tf.keras.layers.Conv2D`
