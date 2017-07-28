// file to just try out stuff
// mostly used for printf debugging (first iteration)

#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"

// cehck that I got the dimensions in `add_inplace` right
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

    /*
    TensorDesc input(32, 3, 8, 8);
    Model m(input);
    m.emplace<BatchNorm>();

    m.init_forward();
    for (int i = 0; i < 10; ++i) {
        m.forward();
    }
    */
    check_add();

    miopenDestroy(mio::handle());
    return 0;
}
