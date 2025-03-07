#ifndef _NN_H_
#define _NN_H_

// TODO: make sure nn.h/gym.h is compilable with C++ compiler
// TODO: introduce _NNDEF macro for every definition of nn.h

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifndef _NN_MALLOC
#include <stdlib.h>
#define _NN_MALLOC malloc
#endif // _NN_MALLOC

#ifndef _NN_ASSERT
#include <assert.h>
#define _NN_ASSERT assert
#endif // _NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef enum {
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN,
} Act;

float rand_float(void);

float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);

// Dispatch to the corresponding activation function
float actf(float x, Act act);

// Derivative of the activation function based on its value
float dactf(float y, Act act);

typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *words;
} Region;

// capacity is in bytes, but it can allocate more just to keep things
// word aligned
Region region_alloc_alloc(size_t capacity_bytes);
void *region_alloc(Region *r, size_t size_bytes);
#define region_reset(r) (_NN_ASSERT((r) != NULL), (r)->size = 0)
#define region_occupied_bytes(r) (_NN_ASSERT((r) != NULL), (r)->size*sizeof(*(r)->words))
#define region_save(r) (_NN_ASSERT((r) != NULL), (r)->size)
#define region_rewind(r, s) (_NN_ASSERT((r) != NULL), (r)->size = s)

typedef struct {
    size_t rows;
    size_t cols;
    float *elements;
} Mat;

typedef struct {
    size_t cols;
    float *elements;
} Row;

#define ROW_AT(row, col) (row).elements[col]

Mat row_as_mat(Row row);
#define row_alloc(cols) mat_row(mat_alloc(1, cols), 0)
Row row_slice(Row row, size_t i, size_t cols);
#define row_rand(row, low, high) mat_rand(row_as_mat(row), low, high)
#define row_fill(row, x) mat_fill(row_as_mat(row), x);
#define row_print(row, name, padding) mat_print(row_as_mat(row), name, padding)
#define row_copy(dst, src) mat_copy(row_as_mat(dst), row_as_mat(src))

#define MAT_AT(m, i, j) (m).elements[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Row mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sub(Mat dst, Mat a);
void mat_mul(Mat dst, float x);
void mat_mul_mat(Mat a, Mat b);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_shuffle_rows(Mat m);
Mat mat_transpose(Mat m);
void mat_transpose_dst(Mat dst, Mat m);
void mat_free(Mat m);
void mat_save(Mat m, FILE *file);
Mat mat_load(size_t row, size_t col, FILE *file);
#define MAT_PRINT(m) mat_print(m, #m, 0)

#endif // _NN_H_

#ifdef MATRIX_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.elements = malloc(sizeof(float) * rows * cols);
    _NN_ASSERT(m.elements != NULL);
    return m;
}

Mat mat_transpose(Mat m) {
    Mat dst = mat_alloc(m.cols, m.rows);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(dst, j, i) = MAT_AT(m, i, j);
        }
    }
    return dst;
}

void mat_transpose_dst(Mat dst, Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(dst, j, i) = MAT_AT(m, i, j);
        }
    }
}

void mat_free(Mat m) {
    free(m.elements);
}

void mat_dot(Mat dst, Mat a, Mat b) {
    _NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    _NN_ASSERT(dst.rows == a.rows);
    _NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Row mat_row(Mat m, size_t row) {
    return (Row) {
        .cols = m.cols,
        .elements = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src)
{
    _NN_ASSERT(dst.rows == src.rows);
    _NN_ASSERT(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    _NN_ASSERT(dst.rows == a.rows);
    _NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sub(Mat dst, Mat a)
{
    _NN_ASSERT(dst.rows == a.rows);
    _NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) -= MAT_AT(a, i, j);
        }
    }
}

void mat_mul(Mat dst, float x)
{
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(dst, i, j) * x;
        }
    }
}

void mat_mul_mat(Mat a, Mat b) {
    _NN_ASSERT(a.rows == b.rows);
    _NN_ASSERT(b.cols == a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            MAT_AT(a, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

void mat_shuffle_rows(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
         size_t j = i + rand()%(m.rows - i);
         if (i != j) {
             for (size_t k = 0; k < m.cols; ++k) {
                 float t = MAT_AT(m, i, k);
                 MAT_AT(m, i, k) = MAT_AT(m, j, k);
                 MAT_AT(m, j, k) = t;
             }
         }
    }
}

void mat_save(Mat m, FILE *file) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            fwrite((const void *)&MAT_AT(m, i, j), sizeof(float), 1, file);
        }
    }
}

Mat mat_load(size_t row, size_t col, FILE *file) {
    Mat m = mat_alloc(row, col);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            float value;
            size_t ret = fread((void *)&value, sizeof(float), 1, file);
            (void)ret;
            MAT_AT(m, i, j) = value;
        }
    }
    return m;
}


Region region_alloc_alloc(size_t capacity_bytes)
{
    Region r = {0};

    size_t word_size = sizeof(*r.words);
    size_t capacity_words = (capacity_bytes + word_size - 1)/word_size;

    void *words = _NN_MALLOC(capacity_words*word_size);
    _NN_ASSERT(words != NULL);
    r.capacity = capacity_words;
    r.words = words;
    return r;
}

void *region_alloc(Region *r, size_t size_bytes)
{
    if (r == NULL) return _NN_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1)/word_size;

    _NN_ASSERT(r->size + size_words <= r->capacity);
    if (r->size + size_words > r->capacity) return NULL;
    void *result = &r->words[r->size];
    r->size += size_words;
    return result;
}

Mat row_as_mat(Row row)
{
    return (Mat) {
        .rows = 1,
        .cols = row.cols,
        .elements = row.elements,
    };
}

Row row_slice(Row row, size_t i, size_t cols)
{
    _NN_ASSERT(i < row.cols);
    _NN_ASSERT(i + cols <= row.cols);
    return (Row) {
        .cols = cols,
        .elements = &ROW_AT(row, i),
    };
}

#endif // _NN_IMPLEMENTATION