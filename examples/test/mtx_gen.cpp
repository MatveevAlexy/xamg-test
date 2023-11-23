#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <cmath>

# define M_PI           3.14159265358979323846

int main(int argc, char **argv)
{
    int n = strtol(argv[1], NULL, 10);
    uint32_t nrows = (n + 1) * (n + 1), row_cnt = 0;
    double h = M_PI / n / 2;
    uint32_t nonzeros = 2 * (n + 1) + (n - 1) * (5 * (n - 1) + 2), cnt = 0;
    double *elems = new double[nonzeros];
    uint32_t *row_idx = new uint32_t[nrows + 1];
    uint32_t *col_idx = new uint32_t[nonzeros];
    // for (int i = 0; i <= n; i++) {
    //     elems[cnt] = 1;
    //     col_idx[cnt] = i;
    //     row_idx[row_cnt++] = cnt;
    //     cnt++;
    // }
    // for (int i = 1; i < n; i++) {
    //     elems[cnt] = 1;
    //     col_idx[cnt] = (n + 1) * i;
    //     row_idx[row_cnt++] = cnt;
    //     cnt++;
    //     for (int j = 1; j < n; j++) {
    //         elems[cnt] = -1.0 / h / h;
    //         col_idx[cnt] = (i - 1) * (n + 1) + j;
    //         row_idx[row_cnt++] = cnt;
    //         cnt++;
    //         elems[cnt] = -1.0 / h / h;
    //         col_idx[cnt] = i * (n + 1) + j - 1;
    //         cnt++;
    //         elems[cnt] = 4.0 / h / h;
    //         col_idx[cnt] = i * (n + 1) + j;
    //         cnt++;
    //         elems[cnt] = -1.0 / h / h;
    //         col_idx[cnt] = i * (n + 1) + 1 + j;
    //         cnt++;
    //         elems[cnt] = -1.0 / h / h;
    //         col_idx[cnt] = (i + 1) * (n + 1) + j;
    //         cnt++;
    //     }
    //     elems[cnt] = 1;
    //     col_idx[cnt] = (n + 1) * i + n;
    //     row_idx[row_cnt++] = cnt;
    //     cnt++;
    // }
    // for (int i = 0; i <= n; i++) {
    //     elems[cnt] = 1;
    //     col_idx[cnt] = n * (n + 1) + i;
    //     row_idx[row_cnt++] = cnt;
    //     cnt++;
    // }
    // row_idx[row_cnt++] = cnt;
    // if (cnt != nonzeros) {
    //     std::cout << cnt << ' ' << nonzeros << std::endl;
    // }
    // if (row_cnt != nrows + 1) {
    //     std::cout << row_cnt << ' ' << nrows + 1 << std::endl;
    // }

    for (int i = 0; i < nonzeros; i++) {
        if (i <= n+1) {
            elems[i] = 1;
            col_idx[i] = i;
        } else if(i >= nonzeros - n - 1) {
            elems[i] = 1;
            col_idx[i] = i - (n - 1) * (4 * (n - 1));
        } else {
            int j = (i-(n+1)) % (5*(n-1) + 2);
            if (j == 0 || j == 5 * (n - 1) + 1) {
                elems[i] = 1;
                col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1;
                if (j == 5 * (n - 1) + 1) {
                    col_idx[i] += n;
                }
            } else {
                if ((j - 1) % 5 == 2) {
                    elems[i] = 4.0 / h / h;
                    col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + 1;
                } else {
                    elems[i] = -1.0 / h / h;
                    if ((j - 1) % 5 == 0) {
                        col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 - n;
                    } else if ((j - 1) % 5 == 1) {
                        col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5;
                    } else if ((j - 1) % 5 == 3) {
                        col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + 2;
                    } else {
                        col_idx[i] = (i-(n+1)) / (5*(n-1) + 2) * (n + 1) + n + 1 + (j - 1) / 5 + n + 2;
                    }
                }
            }
        }
    }

    for (int i = 0; i < nrows + 1; i++) {
        if (i <= n+1) {
            row_idx[i] = i;
        } else if (i >= nrows - n - 1) {
            row_idx[i] = i + (n-1)*(n-1)*4;
        } else {
            int j = (i-(n+1)) / (n+1);
            if ((i-(n+1)) % (n+1) <= 1) {
                row_idx[i] = i + j * (n - 1) * 4;
            } else {
                row_idx[i] = i + j * (n - 1) * 4 + ((i-(n+1)) % (n+1) - 1) * 4;
            }
        }
    }

    FILE *f = fopen("mtx_Paus.bin", "w");
    fwrite(&nrows, sizeof(uint32_t), 1, f);
    fwrite(&nonzeros, sizeof(uint32_t), 1, f);
    fwrite(row_idx, sizeof(uint32_t), nrows + 1, f);
    fwrite(col_idx, sizeof(uint32_t), nonzeros, f);
    fwrite(elems, sizeof(double), nonzeros, f);
    double *vec = new double[nrows];
    // for (int i = 0; i < n + 1; i++) {
    //     for (int j = 0; j < n + 1; j++) {
    //         if (i * j == 0 || i == n || j == n) {
    //             vec[i * (n + 1) + j] = cos(h * i) * sin(h * j);
    //         } else {
    //             vec[i * (n + 1) + j] = cos(h * i) * sin(h * j) * (2.0);
    //         }
    //     }
    // }
    for (int i = 0; i < n + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            if (i * j == 0 || i == n || j == n) {
                vec[i * (n + 1) + j] = cos(h * i) * sin(h * j);
            } else {
                vec[i * (n + 1) + j] = cos(h * i) * sin(h * j) * (2.0);
            }
        }
    }
    fwrite(vec, sizeof(double), nrows, f);
    for (int i = 0; i < nrows; i++) {
        vec[i] = 1;
    }

    fwrite(vec, sizeof(double), nrows, f);
    fclose(f);
    delete [] elems;
    delete [] row_idx;
    delete [] col_idx;
    delete [] vec;
    return 0;
}
