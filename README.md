# nets.h
Single header neural network library in C

```C
#include <stdio.h>

#define NETS_IMPLEMENTATION
#include "nets.h"

static float X[8] = {
    0.0, 0.0, 
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0,
};

static float Y[4] = {
    0.0,
    1.0,
    1.0,
    0.0,
};

#define SIZE 2
#define SAMPLES 4

int main(void) {

    Mat x_train = (Mat){.rows=SAMPLES, .cols=SIZE};
    x_train.elements = X;

    Mat y_train = (Mat){.rows=SAMPLES, .cols=1};
    y_train.elements = Y;

    NN nn = nn_create(
        6,
        DENSE(2, 3),
        SIGMOID(3),
        DENSE(3, 2),
        SIGMOID(2),
        DENSE(2, 1),
        SIGMOID(1)
    );

    for (size_t epoch = 0; epoch < 10000; epoch++) {
        compute_gradient(&nn, x_train, y_train, 0.80f);
        printf("epoch (%ld/100) cost = %f\n", epoch, nn_cost(&nn, x_train, y_train));
    }

    
    static float test[2] = {0.0, 1.0};
    Mat forward = mat_alloc(2, 1);
    forward.elements = test;
    Mat output = nn_forward(&nn, forward);

    MAT_PRINT(output);
    
    nn_free(&nn);

    return 0;
}
```
