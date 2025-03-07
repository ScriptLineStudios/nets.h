#ifndef NETS_H
#define NETS_H

#include "matrix.h"

static inline float sigmoid(float x) {
    return (1) / (1 + expf(-x));
}

#define C 0.1
static inline float leaky_relu(float x) {
    if (x >= 0) {
        return x;
    }
    else {
        return C * x;
    }
}

typedef enum {
    DENSE_LAYER = 0, SIGMOID_LAYER = 1, RELU_LAYER = 2, CELL_LAYER = 3,
} LayerType;

typedef struct {
    Mat x_train;
    Mat y_train;
} Batch;

typedef struct {
    Batch *batches;
    size_t num_batches;
} Batches;

void print_batches(Batch *batches, size_t num_batches);
void shuffle_batches(Batch *batches, size_t num_batches);
Batches create_batches(Mat x_data, Mat y_data, size_t batch_size);

typedef struct {
    LayerType type;
    void *layer;
    size_t input_size;
    size_t output_size;
} Layer;

typedef struct {
    Mat weights;
    Mat bias;
    size_t input_size;
    size_t output_size;
    Mat inputs;
    Mat weights_gradient;
} DenseLayer;

typedef struct {
    size_t input_size;
    size_t output_size;
    Mat inputs;
} ReluLayer;

typedef struct {
    size_t input_size;
    size_t output_size;
    Mat inputs;
} SigmoidLayer;

typedef struct {
    Layer **layers;
    size_t num_layers;
    Mat *activations;
} NN;

NN nn_new(int num_layers);
NN nn_create(int num_layers, ...);
Mat nn_forward(NN *nn, Mat input);
float nn_cost(NN *nn, Mat x_train, Mat y_train);
void compute_gradient(NN *nn, Mat x_train, Mat y_train, float lr);

void nn_print(NN *nn, bool print_params);

Layer *layer_new(LayerType type, void *layer, size_t input_size, size_t output_size);

Layer *dense_layer_new(size_t input_size, size_t output_size);
void dense_layer_forward(DenseLayer *layer, Mat input, Mat output);
Mat dense_layer_backward(DenseLayer *layer, Mat input, float lr);
DenseLayer *as_dense_layer(Layer *layer);
void dense_free(DenseLayer *layer);
#define DENSE(size_in, size_out) dense_layer_new(size_in, size_out)

Layer *relu_layer_new(size_t input_size);
void relu_layer_forward(ReluLayer *layer, Mat input, Mat output);
Mat relu_layer_backward(ReluLayer *layer, Mat input, float lr);
ReluLayer *as_relu_layer(Layer *layer);
void relu_free(ReluLayer *layer);
#define RELU(size) relu_layer_new(size)

Layer *sigmoid_layer_new(size_t input_size);
void sigmoid_layer_forward(SigmoidLayer *layer, Mat input, Mat output);
Mat sigmoid_layer_backward(SigmoidLayer *layer, Mat input, float lr);
SigmoidLayer *as_sigmoid_layer(Layer *layer);
void sigmoid_free(SigmoidLayer *layer);
#define SIGMOID(size) sigmoid_layer_new(size)

void layer_forward(Layer *layer, Mat input, Mat output);
Mat layer_backward(Layer *layer, Mat input, float lr);
void layer_print(Layer *layer, bool print_params);
void layer_free(Layer *layer);

void dense_layer_save(DenseLayer *layer, FILE *file);
void sigmoid_layer_save(SigmoidLayer *layer, FILE *file);
void relu_layer_save(ReluLayer *layer, FILE *file);
void layer_save(Layer *layer, FILE *file);

void nn_save(NN *nn, const char *filename);
NN nn_load(const char *filename);
void nn_free(NN *nn);

#endif

// END

#ifdef NETS_IMPLEMENTATION

#define MATRIX_IMPLEMENTATION
#include "matrix.h"

#include <stdarg.h>
#include <time.h>

Layer *layer_new(LayerType type, void *layer, size_t input_size, size_t output_size) {
    Layer *ret = malloc(sizeof(Layer));
    ret->type = type;
    ret->layer = layer;
    ret->input_size = input_size;
    ret->output_size = output_size;
    return ret;
}

Layer *dense_layer_new(size_t input_size, size_t output_size) {
    DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
    
    Mat weights = mat_alloc(output_size, input_size);
    // weights.elements = test_weights[calls];
    mat_rand(weights, -1, 1);
    layer->weights = weights;

    Mat bias = mat_alloc(output_size, 1);
    // bias.elements = test_bias[calls];
    mat_rand(bias, -1, 1);
    layer->bias = bias; 

    layer->inputs = mat_alloc(1, input_size);

    layer->input_size = input_size;
    layer->output_size = output_size;

    return layer_new(DENSE_LAYER, layer, input_size, output_size);
}

void dense_layer_forward(DenseLayer *layer, Mat input, Mat output) {
    mat_transpose_dst(layer->inputs, input);
    mat_dot(output, layer->weights, input);
    mat_sum(output, layer->bias);
}

Mat dense_layer_backward(DenseLayer *layer, Mat input, float lr) {
    // printf("--- DENSE ---\n");
    // MAT_PRINT(input);
    Mat weights_gradient = mat_alloc(input.rows, layer->inputs.cols);
    mat_dot(weights_gradient, input, layer->inputs);
    Mat input_gradient = mat_alloc(layer->weights.cols, input.cols);
    Mat transposed_weights = mat_transpose(layer->weights);
    mat_dot(input_gradient, transposed_weights, input);

    mat_mul(weights_gradient, lr);
    mat_sub(layer->weights, weights_gradient);

    mat_mul(input, lr);
    mat_sub(layer->bias, input);

    // MAT_PRINT(input_gradient);
    mat_free(weights_gradient);
    mat_free(transposed_weights);
    mat_free(input);

    return input_gradient;
}

DenseLayer *as_dense_layer(Layer *layer) {
    return (DenseLayer *)(layer->layer);
}

void dense_free(DenseLayer *layer) {
    mat_free(layer->weights);
    mat_free(layer->bias);
    mat_free(layer->inputs);
    mat_free(layer->weights_gradient);
}

Layer *relu_layer_new(size_t input_size) {
    ReluLayer *layer = (ReluLayer *)malloc(sizeof(ReluLayer));
    layer->input_size = input_size;
    layer->inputs = mat_alloc(input_size, 1);
    layer->input_size = input_size;
    layer->output_size = input_size;
    return layer_new(RELU_LAYER, layer, input_size, input_size);
}

void relu_layer_forward(ReluLayer *layer, Mat input, Mat output) {
    for (size_t i = 0; i < input.rows; i++) {
        for (size_t j = 0; j < input.cols; j++) {
            MAT_AT(output, i, j) = leaky_relu(MAT_AT(input, i, j));
        }
    }
    mat_copy(layer->inputs, input);
}

Mat relu_layer_backward(ReluLayer *layer, Mat input, float lr) {
    Mat output = mat_alloc(layer->inputs.rows, layer->inputs.cols);
    
    for (size_t i = 0; i < layer->inputs.rows; i++) {
        for (size_t j = 0; j < layer->inputs.cols; j++) {
            MAT_AT(output, i, j) = leaky_relu(MAT_AT(layer->inputs, i, j));
        }
    }

    for (size_t i = 0; i < layer->inputs.rows; i++) {
        for (size_t j = 0; j < layer->inputs.cols; j++) {
            if (MAT_AT(output, i, j) > 0) {
                MAT_AT(output, i, j) = 1;
            }
            else {
                MAT_AT(output, i, j) = C;
            }
        }
    }

    mat_mul_mat(input, output);
    mat_free(output); 
    // MAT_PRINT(input);
    return input;
}

ReluLayer *as_relu_layer(Layer *layer) {
    return (ReluLayer *)(layer->layer);
}

void relu_free(ReluLayer *layer) {
    mat_free(layer->inputs);
}

Layer *sigmoid_layer_new(size_t input_size) {
    SigmoidLayer *layer = (SigmoidLayer *)malloc(sizeof(SigmoidLayer));
    layer->input_size = input_size;
    layer->inputs = mat_alloc(input_size, 1);
    layer->input_size = input_size;
    layer->output_size = input_size;
    return layer_new(SIGMOID_LAYER, layer, input_size, input_size);
}

void sigmoid_layer_forward(SigmoidLayer *layer, Mat input, Mat output) {
    for (size_t i = 0; i < input.rows; i++) {
        for (size_t j = 0; j < input.cols; j++) {
            MAT_AT(output, i, j) = sigmoid(MAT_AT(input, i, j));
        }
    }
    mat_copy(layer->inputs, input);
}

Mat sigmoid_layer_backward(SigmoidLayer *layer, Mat input, float lr) {
    // printf("--- ACTIVATION ---\n");
    // MAT_PRINT(input);
    Mat output = mat_alloc(layer->inputs.rows, layer->inputs.cols);
    
    for (size_t i = 0; i < layer->inputs.rows; i++) {
        for (size_t j = 0; j < layer->inputs.cols; j++) {
            MAT_AT(output, i, j) = sigmoid(MAT_AT(layer->inputs, i, j));
        }
    }

    for (size_t i = 0; i < layer->inputs.rows; i++) {
        for (size_t j = 0; j < layer->inputs.cols; j++) {
            MAT_AT(output, i, j) = MAT_AT(output, i, j) * (1 - MAT_AT(output, i, j));
        }
    }

    mat_mul_mat(input, output);
    mat_free(output); 
    // MAT_PRINT(input);
    return input;
}

SigmoidLayer *as_sigmoid_layer(Layer *layer) {
    return (SigmoidLayer *)(layer->layer);
}

void sigmoid_free(SigmoidLayer *layer) {
    mat_free(layer->inputs);
}

void layer_forward(Layer *layer, Mat input, Mat output) {
    LayerType type = layer->type;
    switch (type) {
        case DENSE_LAYER:
            dense_layer_forward(as_dense_layer(layer), input, output);
            break;
        case SIGMOID_LAYER:
            sigmoid_layer_forward(as_sigmoid_layer(layer), input, output);
            break;
        case RELU_LAYER:
            relu_layer_forward(as_relu_layer(layer), input, output);
            break;
        default:
            printf("unknown! %d\n", type);
            exit(1);
    }
}

Mat layer_backward(Layer *layer, Mat input, float lr) {
    LayerType type = layer->type;
    switch (type) {
        case DENSE_LAYER:
            return dense_layer_backward(as_dense_layer(layer), input, lr);
        case SIGMOID_LAYER:
            return sigmoid_layer_backward(as_sigmoid_layer(layer), input, lr);
        case RELU_LAYER:
            return relu_layer_backward(as_relu_layer(layer), input, lr);
        default:
            printf("unknown! %d\n", type);
            exit(1);
    }
}

void layer_print(Layer *layer, bool print_params) {
    LayerType type = layer->type;
    switch (type) {
        case DENSE_LAYER:
            printf("Dense Layer: %ld %ld\n", layer->input_size, layer->output_size);
            if (print_params) {
                MAT_PRINT(as_dense_layer(layer)->weights);
                MAT_PRINT(as_dense_layer(layer)->bias);
            }
            return;
        case SIGMOID_LAYER:
            printf("Sigmoid Layer: %ld\n", layer->input_size);
            return;
        case RELU_LAYER:
            printf("Relu Layer: %ld\n", layer->input_size);
            return;
        default:
            printf("unknown! %d\n", type);
            exit(1);
    }
}

void dense_layer_save(DenseLayer *layer, FILE *file) {
    LayerType type = DENSE_LAYER;
    fwrite((const void *)&type, sizeof(LayerType), 1, file);
    fwrite((const void *)&layer->input_size, sizeof(size_t), 1, file);
    fwrite((const void *)&layer->output_size, sizeof(size_t), 1, file);
    mat_save(layer->weights, file);
    mat_save(layer->bias, file);
}

void sigmoid_layer_save(SigmoidLayer *layer, FILE *file) {
    LayerType type = SIGMOID_LAYER;
    fwrite((const void *)&type, sizeof(LayerType), 1, file);
    fwrite((const void *)&layer->input_size, sizeof(size_t), 1, file);
    fwrite((const void *)&layer->output_size, sizeof(size_t), 1, file);
}

void relu_layer_save(ReluLayer *layer, FILE *file) {
    LayerType type = RELU_LAYER;
    fwrite((const void *)&type, sizeof(LayerType), 1, file);
    fwrite((const void *)&layer->input_size, sizeof(size_t), 1, file);
    fwrite((const void *)&layer->output_size, sizeof(size_t), 1, file);
}

void layer_save(Layer *layer, FILE *file) {
    LayerType type = layer->type;
    switch (type) {
        case DENSE_LAYER:
            dense_layer_save(as_dense_layer(layer), file);
            return;
        case SIGMOID_LAYER:
            sigmoid_layer_save(as_sigmoid_layer(layer), file);
            return;
        case RELU_LAYER:
            relu_layer_save(as_relu_layer(layer), file);
            return;
        default:
            printf("unknown! %d\n", type);
            exit(1);
    }
}

void layer_free(Layer *layer) {
    LayerType type = layer->type;
    switch (type) {
        case DENSE_LAYER:
            dense_free(as_dense_layer(layer));
            break;
        case SIGMOID_LAYER:
            sigmoid_free(as_sigmoid_layer(layer));
            break;
        case RELU_LAYER:
            relu_free(as_relu_layer(layer));
            break;
        default:
            printf("unknown! %d\n", type);
            exit(1);
    }
    free(layer->layer);
    free(layer);
}

NN nn_new(int num_layers) {
    NN nn;
    nn.num_layers = num_layers;
    nn.layers = malloc(sizeof(Layer *) * num_layers);
    nn.activations = malloc(sizeof(Mat) * (num_layers+1));
    return nn;
}

NN nn_create(int num_layers, ...) {
    va_list ptr;
    va_start(ptr, num_layers);
    
    NN nn = nn_new(num_layers);

    for (int i = 0; i < num_layers; i++) {
        Layer *layer = va_arg(ptr, Layer *);
        nn.layers[i] = layer;
    }

    for (size_t i = 0; i < nn.num_layers; i++) {
        nn.activations[i] = mat_alloc(nn.layers[i]->input_size, 1);
    }
    nn.activations[nn.num_layers] = mat_alloc(nn.layers[nn.num_layers - 1]->output_size, 1);

    va_end(ptr);
    return nn;
}

Mat nn_forward(NN *nn, Mat input) {
    mat_copy(nn->activations[0], input);

    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *current_layer = nn->layers[i];
        layer_forward(current_layer, nn->activations[i], nn->activations[i+1]);
    }

    Mat output = mat_alloc(nn->activations[nn->num_layers].rows, nn->activations[nn->num_layers].cols);
    mat_copy(output, nn->activations[nn->num_layers]);
    return output;
}

float sample_cost(Mat y_true, Mat y_pred) {
    assert(y_true.rows == y_pred.rows);
    assert(y_true.cols == y_pred.cols == 1);

    for (int i = 0; i < y_true.rows; i++) {
        MAT_AT(y_pred, i, 0) = pow(MAT_AT(y_true, i, 0) - MAT_AT(y_pred, i, 0), 2);
    }

    float average = 0.0;
    for (int i = 0; i < y_true.rows; i++) {
        average += MAT_AT(y_pred, i, 0);
    }
    return average / y_true.rows;
}

Mat gsample_cost(Mat y_true, Mat y_pred) {
    assert(y_true.rows == y_pred.rows);
    assert(y_true.cols == y_pred.cols == 1);
    for (int i = 0; i < y_true.rows; i++) {
        MAT_AT(y_pred, i, 0) = 2 * (MAT_AT(y_pred, i, 0) - MAT_AT(y_true, i, 0));
    }
    for (int i = 0; i < y_true.rows; i++) {
        MAT_AT(y_pred, i, 0) = MAT_AT(y_pred, i, 0) / (y_true.rows * y_true.cols);
    }

    return y_pred;
}

Mat sample_at(Mat x_train, size_t index) {
    return mat_transpose(row_as_mat(mat_row(x_train, index)));
}

float nn_cost(NN *nn, Mat x_train, Mat y_train) {
    float cost = 0.0;
    for (size_t i = 0; i < x_train.rows; i++) {
        Mat X = sample_at(x_train, i);
        Mat Y = sample_at(y_train, i);

        Mat output = nn_forward(nn, X);
        
        cost += sample_cost(Y, output);
        mat_free(X);
        mat_free(Y);
        mat_free(output);
    }
    return cost / x_train.rows;
}

void compute_gradient(NN *nn, Mat x_train, Mat y_train, float lr) {
    for (int i = 0; i < x_train.rows; i++) {
        Mat X = mat_transpose(row_as_mat(mat_row(x_train, i)));
        Mat Y = mat_transpose(row_as_mat(mat_row(y_train, i)));

        Mat output = nn_forward(nn, X);

        Mat gradient = gsample_cost(Y, output);
        for (int j = nn->num_layers - 1; j >= 0; j--) {
            Layer *layer = nn->layers[j];
            gradient = layer_backward(layer, gradient, lr);
        }

        mat_free(X);
        mat_free(Y);
        mat_free(gradient);
    }
}

void print_batches(Batch *batches, size_t num_batches) {
    for (size_t i = 0; i < num_batches; i++) {
        printf("BATCH %ld\n", i);
        MAT_PRINT(batches[i].x_train);
        MAT_PRINT(batches[i].y_train);
    }
}

void shuffle_batches(Batch *batches, size_t num_batches) {
    long long seed = time(0);
    for (size_t i = 0; i < num_batches; i++) {
        srand(seed);
        mat_shuffle_rows(batches[i].x_train);
        srand(seed);
        mat_shuffle_rows(batches[i].y_train);
        srand(seed + i);
        int new_index = (rand() % num_batches) - 1;
        Batch temp = batches[new_index];
        batches[new_index] = batches[i];
        batches[i] = temp;
    }
}

Batches create_batches(Mat x_data, Mat y_data, size_t batch_size) {
    assert(x_data.rows == y_data.rows);
    size_t data_length = x_data.rows;
    size_t points_per_batch = (data_length / batch_size);

    Batch *batches = (Batch *)malloc(sizeof(Batch) * points_per_batch);
    size_t index = 0;
    for (size_t i = 0; i < data_length; i += points_per_batch) {
        Mat x = (Mat){
            .rows=points_per_batch,
            .cols=x_data.cols,
            .elements=&MAT_AT(x_data, i, 0)
        };
        Mat y = (Mat){
            .rows=points_per_batch,
            .cols=y_data.cols,
            .elements=&MAT_AT(y_data, i, 0)
        };
        Batch batch = (Batch){.x_train=x, .y_train=y};
        batches[index] = batch;
        index++;
    }
    shuffle_batches(batches, index-1);

    return (Batches){.batches=batches, .num_batches=index-1};
}

void nn_save(NN *nn, const char *filename) {
    printf("[INFO]: Saving neural network to %s\n", filename);
    FILE *file = fopen(filename, "wb");

    const long magic = 0x4245414E530A;
    fwrite((const void *)&magic, sizeof(long), 1, file);
    fwrite((const void *)&nn->num_layers, sizeof(size_t), 1, file);

    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers[i];
        layer_save(layer, file);
    }

    fclose(file);
}

#define UNUSED(x) (void)x

NN nn_load(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "[ERROR]: Unable to open: %s\n", filename);
        exit(1);
    }

    long magic;
    size_t ret = fread((void *)&magic, sizeof(long), 1, file);
    UNUSED(ret);
    assert(magic == 0x4245414E530A);
    
    size_t num_layers;
    ret = fread((void *)&num_layers, sizeof(size_t), 1, file);
    UNUSED(ret);

    NN nn = nn_new(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        LayerType type;
        ret = fread((void *)&type, sizeof(LayerType), 1, file);
        UNUSED(ret);
        
        Layer *layer;
        switch (type) {
            case DENSE_LAYER: {
                size_t input_size;
                size_t output_size;

                ret = fread((void *)&input_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                ret = fread((void *)&output_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                Mat weights = mat_load(output_size, input_size, file);
                Mat bias = mat_load(output_size, 1, file);

                layer = dense_layer_new(input_size, output_size);
                mat_free(as_dense_layer(layer)->weights);
                mat_free(as_dense_layer(layer)->bias);
                as_dense_layer(layer)->weights = weights;
                as_dense_layer(layer)->bias = bias;

                break;
            }
            case SIGMOID_LAYER: {
                size_t input_size;
                size_t output_size;

                ret = fread((void *)&input_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                ret = fread((void *)&output_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                layer = sigmoid_layer_new(input_size);
                break;
            }
            case RELU_LAYER: {
                size_t input_size;
                size_t output_size;

                ret = fread((void *)&input_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                ret = fread((void *)&output_size, sizeof(size_t), 1, file);
                UNUSED(ret);

                layer = relu_layer_new(input_size);
                break;
            }
            default:
                assert(false);
        }

        nn.layers[i] = layer;
    }

    for (size_t i = 0; i < nn.num_layers; i++) {
        nn.activations[i] = mat_alloc(nn.layers[i]->input_size, 1);
    }
    nn.activations[nn.num_layers] = mat_alloc(nn.layers[nn.num_layers - 1]->output_size, 1);

    fclose(file);
    return nn;
}

void nn_print(NN *nn, bool print_params) {
    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers[i];
        layer_print(layer, print_params);
    }
}

void nn_free(NN *nn) {
    for (size_t i = 0; i < nn->num_layers; i++) {
        Layer *layer  = nn->layers[i];
        layer_free(layer);
    }
    for (size_t i = 0; i < nn->num_layers; i++) {
        mat_free(nn->activations[i]);
    }
    mat_free(nn->activations[nn->num_layers]);
    free(nn->activations);
    free(nn->layers);
}

#endif //NETS_IMPLEMENTATION

