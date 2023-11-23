#include <iostream>
#include <xamg/xamg_headers.h>
#include <xamg/xamg_types.h>
#include <cmath>


#include <string>

#include <cstdio>


#ifndef FP_TYPE
#define FP_TYPE float64_t
#endif

#ifdef ITAC_TRACE
#include <VT.h>
#endif


# define M_PI           3.14159265358979323846


template <typename T>
void row_gen(XAMG::vector::vector &vec, const uint64_t row_offset, uint64_t n) {
    auto vec_ptr_row = vec.get_aligned_ptr<T>();
    int nrows = (n + 1) * (n + 1);
    for (size_t i = row_offset; i < vec.size * vec.nv + row_offset; i++) {
        if (i <= n+1) {
            vec_ptr_row[i - row_offset] = i;
        } else if (i >= nrows - n - 1) {
            vec_ptr_row[i - row_offset] = i + (n-1)*(n-1)*4;
        } else {
            size_t j = (i-(n+1)) / (n+1);
            if ((i-(n+1)) % (n+1) <= 1) {
                vec_ptr_row[i - row_offset] = i + j * (n - 1) * 4;
            } else {
                vec_ptr_row[i - row_offset] = i + j * (n - 1) * 4 + ((i-(n+1)) % (n+1) - 1) * 4;
            }
        }
    }
    vec.if_initialized = true;
    vec.if_zero = false;
}

template <typename T, typename F>
void val_col_gen(XAMG::vector::vector &val, XAMG::vector::vector &col, const uint64_t nnz_offset, uint64_t n) {
    auto vec_ptr_col = col.get_aligned_ptr<T>();
    auto vec_ptr_val = val.get_aligned_ptr<F>();
    double h = M_PI / n / 2;
    int nonzeros = 2 * (n + 1) + (n - 1) * (5 * (n - 1) + 2);
    if (val.size != col.size) {
        std::cout << val.size << ' ' << col.size << std::endl;
    }
    for (size_t i = nnz_offset; i < col.size * col.nv + nnz_offset; i++) {
        if (i <= n+1) {
            vec_ptr_val[i- nnz_offset] = 1;
            vec_ptr_col[i- nnz_offset] = i;
        } else if(i >= nonzeros - n - 1) {
            vec_ptr_val[i- nnz_offset] = 1;
            vec_ptr_col[i- nnz_offset] = i - (n - 1) * (4 * (n - 1));
        } else {
            size_t j = (i-(n+1)) % (5*(n-1) + 2);
            if (j == 0 || j == 5 * (n - 1) + 1) {
                vec_ptr_val[i- nnz_offset] = 1;
                vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1;
                if (j == 5 * (n - 1) + 1) {
                    vec_ptr_col[i- nnz_offset] += n;
                }
            } else {
                if ((j - 1) % 5 == 2) {
                    vec_ptr_val[i- nnz_offset] = 4.0 / h / h;
                    vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + 1;
                } else {
                    vec_ptr_val[i- nnz_offset] = -1.0 / h / h;
                    if ((j - 1) % 5 == 0) {
                        vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 - n;
                    } else if ((j - 1) % 5 == 1) {
                        vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5;
                    } else if ((j - 1) % 5 == 3) {
                        vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + 2;
                    } else {
                        vec_ptr_col[i- nnz_offset] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + n + 2;
                    }
                }
            }
        }
    }
    val.if_initialized = true;
    val.if_zero = false;
    col.if_initialized = true;
    col.if_zero = false;
}


template <typename R, typename C, typename F>
void mtx_gen(XAMG::vector::vector &row, XAMG::vector::vector &val, XAMG::vector::vector &col, const uint64_t row_offset, uint64_t n) {
    auto vec_ptr_row = row.get_aligned_ptr<R>();
    auto vec_ptr_col = col.get_aligned_ptr<C>();
    auto vec_ptr_val = val.get_aligned_ptr<F>();
    double h = M_PI / n / 2;

    size_t val_idx = 0, row_idx = 0, val_ptr = 0;

    for (size_t i = 0; i <= n; i++) {
        if (row_idx >= row_offset && row_idx < row.size * row.nv + row_offset - 1) {
            vec_ptr_val[val_ptr] = 1;
            vec_ptr_col[val_ptr] = i;
            val_ptr++;
            vec_ptr_row[row_idx - row_offset] = val_idx;
        } else if (row_idx == row.size * row.nv + row_offset - 1) {
            vec_ptr_row[row_idx - row_offset] = val_idx;
        }
        row_idx++;
        val_idx++;
    }
    for (size_t i = 1; i < n; i++) {
        if (row_idx >= row_offset && row_idx < row.size * row.nv + row_offset - 1) {
            vec_ptr_val[val_ptr] = 1;
            vec_ptr_col[val_ptr] = (n + 1) * i;
            val_ptr++;
            vec_ptr_row[row_idx - row_offset] = val_idx;
        } else if (row_idx == row.size * row.nv + row_offset - 1) {
            vec_ptr_row[row_idx - row_offset] = val_idx;
        }
        row_idx++;
        val_idx++;
        for (size_t j = 1; j < n; j++) {
            if (row_idx >= row_offset && row_idx < row.size * row.nv + row_offset - 1) {
                vec_ptr_val[val_ptr] = -1.0;
                vec_ptr_col[val_ptr] = (i - 1) * (n + 1) + j;
                vec_ptr_row[row_idx - row_offset] = val_idx;
                val_ptr++;
                vec_ptr_val[val_ptr] = -1.0;
                vec_ptr_col[val_ptr] = i * (n + 1) + j - 1;
                val_ptr++;
                vec_ptr_val[val_ptr] = 4.0;
                vec_ptr_col[val_ptr] = i * (n + 1) + j;
                val_ptr++;
                vec_ptr_val[val_ptr] = -1.0;
                vec_ptr_col[val_ptr] = i * (n + 1) + 1 + j;
                val_ptr++;
                vec_ptr_val[val_ptr] = -1.0;
                vec_ptr_col[val_ptr] = (i + 1) * (n + 1) + j;
                val_ptr++;
            } else if (row_idx == row.size * row.nv + row_offset - 1) {
                vec_ptr_row[row_idx - row_offset] = val_idx;
            }
            row_idx++;
            val_idx += 5;
        }
        if (row_idx >= row_offset && row_idx < row.size * row.nv + row_offset - 1) {
            vec_ptr_val[val_ptr] = 1;
            vec_ptr_col[val_ptr] = (n + 1) * i + n;
            val_ptr++;
            vec_ptr_row[row_idx - row_offset] = val_idx;
        } else if (row_idx == row.size * row.nv + row_offset - 1) {
            vec_ptr_row[row_idx - row_offset] = val_idx;
        }
        row_idx++;
        val_idx++;
    }
    for (size_t i = 0; i <= n; i++) {
        if (row_idx >= row_offset && row_idx < row.size * row.nv + row_offset - 1) {
            vec_ptr_val[val_ptr] = 1;
            vec_ptr_col[val_ptr] = n * (n + 1) + i;
            val_ptr++;
            vec_ptr_row[row_idx - row_offset] = val_idx;
        } else if (row_idx == row.size * row.nv + row_offset - 1) {
            vec_ptr_row[row_idx - row_offset] = val_idx;
        }
        row_idx++;
        val_idx++;
    }
    if (row_idx == row.size * row.nv + row_offset - 1) {
        vec_ptr_row[row_idx - row_offset] = val_idx;
    }

    val.if_initialized = true;
    val.if_zero = false;
    col.if_initialized = true;
    col.if_zero = false;
    row.if_initialized = true;
    row.if_zero = false;
}


template <class MATRIX_TYPE, uint16_t NV>
void my_generate_system(MATRIX_TYPE &mat, XAMG::vector::vector &x, XAMG::vector::vector &b, uint64_t n) {
    uint32_t row_cnt = 0;
    uint32_t i32_nrows = (n + 1) * (n + 1);
    double h = M_PI / n / 2;
    uint32_t i32_nonzeros = 2 * (n + 1) + (n - 1) * (5 * (n - 1) + 2), cnt = 0;

    using FP = typename MATRIX_TYPE::float_type;
    using ROW_IDX_TYPE = typename MATRIX_TYPE::row_idx_type;
    using COL_IDX_TYPE = typename MATRIX_TYPE::col_idx_type;

    

    uint64_t block_size = i32_nrows / id.gl_nprocs;
    uint64_t block_offset = block_size * id.gl_proc;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = i32_nrows - block_offset;

    uint32_t i1, i2;

    if (block_offset <= n+1) {
        i1 = block_offset;
    } else if (block_offset >= i32_nrows - n - 1) {
        i1 = block_offset + (n-1)*(n-1)*4;
    } else {
        int j = (block_offset-(n+1)) / (n+1);
        if ((block_offset-(n+1)) % (n+1) <= 1) {
            i1 = block_offset + j * (n - 1) * 4;
        } else {
            i1 = block_offset + j * (n - 1) * 4 + ((block_offset-(n+1)) % (n+1) - 1) * 4;
        }
    }

    if ((block_offset + block_size) <= n+1) {
        i2 = (block_offset + block_size);
    } else if ((block_offset + block_size) >= i32_nrows - n - 1) {
        i2 = (block_offset + block_size) + (n-1)*(n-1)*4;
    } else {
        int j = ((block_offset + block_size)-(n+1)) / (n+1);
        if (((block_offset + block_size)-(n+1)) % (n+1) <= 1) {
            i2 = (block_offset + block_size) + j * (n - 1) * 4;
        } else {
            i2 = (block_offset + block_size) + j * (n - 1) * 4 + (((block_offset + block_size)-(n+1)) % (n+1) - 1) * 4;
        }
    }

    uint64_t block_nonzeros = i2 - i1;

    mat.nrows = block_size;
    mat.block_nrows = block_size;
    mat.ncols = i32_nrows;
    mat.block_ncols = i32_nrows;
    mat.block_row_offset = block_offset;
    mat.block_col_offset = 0;
    mat.nonzeros = block_nonzeros;

    mat.alloc();

    uint64_t row_offset = mat.block_row_offset;

    // row_gen<ROW_IDX_TYPE>(mat.row, row_offset, n);

    mtx_gen<ROW_IDX_TYPE, COL_IDX_TYPE, FP>(mat.row, mat.val, mat.col, row_offset, n);
    auto row_ptr = mat.row.template get_aligned_ptr<ROW_IDX_TYPE>();
    uint64_t nnz_offset = row_ptr[0];

    // val_col_gen<COL_IDX_TYPE, FP>(mat.val, mat.col, nnz_offset, n);

    for (uint64_t i = 0; i < mat.row.size; ++i)
        row_ptr[i] -= nnz_offset;
    
    row_offset = mat.block_row_offset * NV;

    x.alloc<FP>(mat.nrows, NV);
    b.alloc<FP>(mat.nrows, NV);

    x.ext_offset = b.ext_offset = block_offset;

    auto vec_ptr_b = b.get_aligned_ptr<FP>();
    for (size_t i = 0; i < b.size * b.nv; i++) {
        if ((row_offset + i) / (n + 1) == 0 || (row_offset + i) % (n + 1) == 0 ||
                (row_offset + i) / (n + 1) == n || (row_offset + i) % (n + 1) == n) {
            vec_ptr_b[i] = cos(h * ((row_offset + i) / (n + 1))) * sin(h * ((row_offset + i) % (n + 1)));
        } else {
            vec_ptr_b[i] = cos(h * ((row_offset + i) / (n + 1))) * sin(h * ((row_offset + i) % (n + 1))) * 2.0 * h * h;
        }
    }
    b.if_initialized = true;
    b.if_zero = false;

    auto vec_ptr_x = x.get_aligned_ptr<FP>();
    for (size_t i = 0; i < x.size * x.nv; i++) {
        vec_ptr_x[i] = 1.0;
    }
    x.if_initialized = true;
    x.if_zero = false;


}


template <typename F>
void my_system(args_parser &parser, XAMG::matrix::matrix &m, XAMG::vector::vector &x,
                        XAMG::vector::vector &b, uint64_t n, const bool graph_reordering = false,
                        const bool save_pattern = false) {
    using matrix_t = XAMG::matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>;
    auto sh_mat_csr = std::make_shared<matrix_t>();
    auto sh_x0 = std::make_shared<XAMG::vector::vector>();
    auto sh_b0 = std::make_shared<XAMG::vector::vector>();

    my_generate_system<matrix_t, NV>(*sh_mat_csr, *sh_x0, *sh_b0, n);
    

    auto part = XAMG::part::make_partitioner(sh_mat_csr->nrows);
    
    XAMG::vector::construct_distributed<F, NV>(part, *sh_b0, b);
    MPI_Barrier(MPI_COMM_WORLD);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_x0, x);
    MPI_Barrier(MPI_COMM_WORLD);
    XAMG::matrix::construct_distributed<matrix_t>(part, *sh_mat_csr, m);
    MPI_Barrier(MPI_COMM_WORLD);
}
