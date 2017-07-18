

#include <assert.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>


#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


/* TODO:
 * - [ ] create AlexNet class
 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors
 * - [x] Make `Model` take input and output tensors in forward(), backward()
 * - [ ] Collect total and average times per layer
 * - [ ] implement and benchmark ResNet
 */


void alexNet() {
    TensorDesc input_dim(128, 3, 224, 224);

    Sequential features(input_dim);
    /* features */
    features.addConv(64, 11, 2, 4);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(192, 5, 2, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(384, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);

    DEBUG("Dims after Features: " << features.getOutputDesc());

    /* classifier */
    Sequential classifier(features.getOutputDesc());
    // TODO Dropout
    classifier.reshape(128, 256 * 6 * 6, 1, 1);
    classifier.addLinear(4096);
    classifier.addReLU();
    // TODO: Dropout
    classifier.addLinear(4096);
    classifier.addReLU();
    classifier.addLinear(1000);

    Model m(input_dim);
    m.add(features);
    m.add(classifier);

    BenchmarkLogger::new_session("alex_net");
    BenchmarkLogger::benchmark(m);
}

void check_add() {
    Tensor x(2, 2, 1, 1);
    x.fromHost({3, 4, 2, 1});
    x.print_data();

    Tensor y(2, 2, 1, 1);
    y.fromHost({-3, .15, 2, 5});
    y.print_data();

    add_inplace(x, y);
    x.print_data();
}

int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    alexNet();

    miopenDestroy(mio::handle());
    return 0;
}
