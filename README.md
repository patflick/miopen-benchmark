# MIOpen Benchmarks

### Benchmarks for MIOpen

- AlexNet
- ResNet 34, 50, 101
- Layerwise benchmarks

### Tools

- Benchmarking class to log kernel runtime and hardware details (temp, clock) to `tsv` and log files
- `gputop`: monitor temperature, engine clock, memory clock, fan speed
- `summarize.sql`: summarize layer-wise `.tsv` benchmarking logs to find the most time consuming layers


### Building:
```
make
```

### Running all benchmarks
```
make benchmark
```
Take a look into the `Makefile` or `.cpp` sources for more details.

