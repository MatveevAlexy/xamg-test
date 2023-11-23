#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>

int main()
{
    FILE *f = fopen("mtx_Paus.bin", "r");
    uint32_t nzer, nrow;
    fread(&nrow, sizeof(uint32_t), 1, f);
    std::cout << nrow << std::endl;
    fread(&nzer, sizeof(uint32_t), 1, f);
    std::cout << nzer << std::endl;
    uint32_t *row_ids = new uint32_t[nrow+1], *col_ids = new uint32_t[nzer];
    double *el = new double[nzer], *b = new double[nrow];
    fread(row_ids, sizeof(uint32_t), nrow+1, f);
    fread(col_ids, sizeof(uint32_t), nzer, f);
    fread(el, sizeof(double), nzer, f);
    fread(b, sizeof(double), nrow, f);

    for (int i = 0; i < nrow; i++) {
        std::cout << i << ": ";
        for (int j = row_ids[i]; j < row_ids[i + 1]; j++) {
            std::cout << '(' << col_ids[j] << ',' << el[j] << ')' << ' ';
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < nrow; i++) {
        std::cout << b[i] << std::endl;
    }

    // for (int j = 0; j < nrow + 1; j++) {
    //     fread(&i, sizeof(uint32_t), 1, f);
    //     //std::cout << i << ' ';
    // }
    // std::cout << std::endl;
    // for (int j = 0; j < nzer; j++) {
    //     fread(&i, sizeof(uint32_t), 1, f);
    //     //std::cout << i << ' ';
    // }
    // std::cout << std::endl;
    // double k;
    // for (int j = 0; j < nzer; j++) {
    //     fread(&k, sizeof(double), 1, f);
    //     //std::cout << k << ' ';
    // }
    //std::cout << std::endl;
    fclose(f);
    return 0;
}