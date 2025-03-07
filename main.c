#include <stdio.h>

#define NETS_IMPLEMENTATION
#include "nets.h"

#include "mnist_x.h"
#include "mnist_y.h"

#define SAMPLES 1

// static float X[8] = {
    // 0.0, 0.0, 
    // 1.0, 0.0,
    // 0.0, 1.0,
    // 1.0, 1.0,
// };
#define SIZE (14 * 14)

// static float Y[4] = {
    // 0.0,
    // 1.0,
    // 1.0,
    // 0.0,
// };

int main(void) {
    NN nn = nn_load("nn5.network");
    
    static float test[SIZE] = {0};
    for (int i = 0; i < SIZE; i++) {
        test[i] = X[i];
        printf("%f, ", test[i]);
    }
    printf("\n");
    Mat forward = mat_alloc(SIZE, 1);
    forward.elements = test;

    Mat output = nn_forward(&nn, forward);

    MAT_PRINT(output);

    return 0;
//     Mat x_train = (Mat){.rows=SAMPLES, .cols=SIZE};
//     x_train.elements = X;

//     Mat y_train = (Mat){.rows=SAMPLES, .cols=SIZE};
//     y_train.elements = Y;

//     srand(time(0));
//     NN nn = nn_create(
//         10,
//         DENSE(SIZE, 128),
//         SIGMOID(128),
//         DENSE(128, 128),
//         SIGMOID(128),
//         DENSE(128, 128),
//         SIGMOID(128),
//         DENSE(128, 32),
//         SIGMOID(32),
//         DENSE(32, SIZE),
//         SIGMOID(SIZE)
//     );

//     for (size_t epoch = 0; epoch < 10000; epoch++) {
//         compute_gradient(&nn, x_train, y_train, 0.80f);
//         printf("epoch (%ld/100) cost = %f\n", epoch, nn_cost(&nn, x_train, y_train));
//     }

//     nn_save(&nn, "nn5.network");
//     return 0;

//     static float test[2] = {1.0, 1.0};
//     Mat forward = mat_alloc(2, 1);
//     forward.elements = test;
//     Mat output = nn_forward(&nn, forward);

//     MAT_PRINT(output);

//     nn_free(&nn);
//     return 0;
}
