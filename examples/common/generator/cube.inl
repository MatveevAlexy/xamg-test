/****************************************************************************
** 
**  Copyright (C) 2019-2021 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#pragma once

//  Test case 0: Isotropic case, a=b=c=1; K=1
//  Test case 1: Anisotropic case, a=0.1; b=1; c=10; K=1
//  Test case 2: Anisotropic case, a=0.001; b=1; c=1; K=1
//  Test case 3: Jumps, a=b=c=1; K=1e3 (core) / 0.01 (corners) / 1 (otherwise)
//  Test case 4: Core, a=0.1; b=1; c=1; K=1e-8 (core) / 1 (otherwise)

template <typename F>
std::vector<F> get_anisotropy_factors(const size_t mode) {
    std::vector<F> aa(3, 1.0);
    if (mode == 1) {
        aa[0] = 0.1;
        aa[2] = 10;
    } else if (mode == 2) {
        aa[0] = 1e-3;
    }
    return aa;
}

template <typename F>
F get_K_coeff(const F xc, const F yc, const F zc, const size_t mode) {
    F val = 1.0;

    if (mode == 3) { // Jumps
        if ((xc >= 0.1) && (xc <= 0.9) && (yc >= 0.1) && (yc <= 0.9) && (zc >= 0.1) && (zc <= 0.9))
            val = 1e3;
        else if (((xc < 0.1) && (yc < 0.1) && (zc < 0.1)) ||
                 ((xc < 0.1) && (yc < 0.1) && (zc > 0.9)) ||
                 ((xc < 0.1) && (yc > 0.9) && (zc < 0.1)) ||
                 ((xc < 0.1) && (yc > 0.9) && (zc > 0.9)) ||
                 ((xc > 0.9) && (yc < 0.1) && (zc < 0.1)) ||
                 ((xc > 0.9) && (yc < 0.1) && (zc > 0.9)) ||
                 ((xc > 0.9) && (yc > 0.9) && (zc < 0.1)) ||
                 ((xc > 0.9) && (yc > 0.9) && (zc > 0.9)))
            val = 1e-2;
    } else if (mode == 4) { // Cube
        if ((xc >= 1. / 3.) && (xc <= 2. / 3.) && (yc >= 1. / 3.) && (yc <= 2. / 3.) &&
            (zc >= 1. / 3.) && (zc <= 2. / 3.))
            val = 1e-8;
    }

    return val;
}

template <typename F, typename I1, typename I2, typename I3, typename I4, uint16_t NV>
bool generate_cube(XAMG::matrix::csr_matrix<F, I1, I2, I3, I4> &mat, XAMG::vector::vector &x,
                   XAMG::vector::vector &b, const uint64_t nx, const uint64_t ny, const uint64_t nz,
                   const size_t config = 0) {
    uint32_t i32_nrows, i32_nonzeros;
    i32_nrows = nx * ny * nz;

    uint64_t block_size = i32_nrows / id.gl_nprocs;
    uint64_t block_offset = block_size * id.gl_proc;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = i32_nrows - block_offset;

    //////////
    //    getting block nonzeros:

    uint64_t block_nonzeros = 0;
    for (uint32_t i = 0; i < block_size; ++i) {
        uint64_t indx = block_offset + i;
        uint64_t ii = indx % nx;
        uint64_t jj = (indx / nx) % ny;
        uint64_t kk = indx / (nx * ny);

        ++block_nonzeros;
        if (kk > 0)
            ++block_nonzeros;
        if (jj > 0)
            ++block_nonzeros;
        if (ii > 0)
            ++block_nonzeros;

        if (ii < nx - 1)
            ++block_nonzeros;
        if (jj < ny - 1)
            ++block_nonzeros;
        if (kk < nz - 1)
            ++block_nonzeros;
    }

    //////////

    mat.nrows = block_size;
    mat.block_nrows = block_size;
    mat.ncols = i32_nrows;
    mat.block_ncols = i32_nrows;
    mat.block_row_offset = block_offset;
    mat.block_col_offset = 0;
    mat.nonzeros = block_nonzeros;

    mat.alloc();
    //    XAMG::out << XAMG::ALLRANKS << XAMG::DBG <<  "Local: "<< mat.nrows << " || " <<
    //    mat.nonzeros << std::endl;

    /////////

    uint64_t row_offset = mat.block_row_offset;

    const auto aa = get_anisotropy_factors<F>(config);

    F hx = 1.0 / nx;
    F hy = 1.0 / ny;
    F hz = 1.0 / nz;

    F q1 = aa[0] / hx / hx;
    F q2 = aa[1] / hy / hy;
    F q3 = aa[2] / hz / hz;

    uint64_t cntr = 0;
    for (uint32_t i = 0; i < block_size; ++i) {
        uint64_t indx = block_offset + i;
        uint64_t ii = indx % nx;
        uint64_t jj = (indx / nx) % ny;
        uint64_t kk = indx / (nx * ny);

        F xc = hx * (ii + 0.5);
        F yc = hy * (jj + 0.5);
        F zc = hz * (kk + 0.5);

        F Kb = get_K_coeff(xc, yc, zc - 0.5 * hz, config);
        F Ks = get_K_coeff(xc, yc - 0.5 * hy, zc, config);
        F Kw = get_K_coeff(xc - 0.5 * hx, yc, zc, config);
        F Ke = get_K_coeff(xc + 0.5 * hx, yc, zc, config);
        F Kn = get_K_coeff(xc, yc + 0.5 * hy, zc, config);
        F Kt = get_K_coeff(xc, yc, zc + 0.5 * hz, config);

        std::vector<F> vals;
        std::vector<uint64_t> inds;

        vals.push_back(0.0);
        inds.push_back(indx);

        if (kk > 0) {
            vals[0] += q3 * Kb;
            vals.push_back(-q3 * Kb);
            inds.push_back(indx - nx * ny);
        } else if (nz > 1)
            vals[0] += 2.0 * q3 * Kb;

        if (jj > 0) {
            vals[0] += q2 * Ks;
            vals.push_back(-q2 * Ks);
            inds.push_back(indx - nx);
        } else if (ny > 1)
            vals[0] += 2.0 * q2 * Ks;

        if (ii > 0) {
            vals[0] += q1 * Kw;
            vals.push_back(-q1 * Kw);
            inds.push_back(indx - 1);
        } else if (nx > 1)
            vals[0] += 2.0 * q1 * Kw;

        if (ii < nx - 1) {
            vals[0] += q1 * Ke;
            vals.push_back(-q1 * Ke);
            inds.push_back(indx + 1);
        } else if (nx > 1)
            vals[0] += 2.0 * q1 * Ke;

        if (jj < ny - 1) {
            vals[0] += q2 * Kn;
            vals.push_back(-q2 * Kn);
            inds.push_back(indx + nx);
        } else if (ny > 1)
            vals[0] += 2.0 * q2 * Kn;

        if (kk < nz - 1) {
            vals[0] += q3 * Kt;
            vals.push_back(-q3 * Kt);
            inds.push_back(indx + nx * ny);
        } else if (nz > 1)
            vals[0] += 2.0 * q3 * Kt;

        mat.upload_row(i, inds, vals, vals.size());
        cntr += vals.size();
    }

    assert(block_nonzeros == cntr);
    ////    std::cout << nonzeros << " " << cntr << "\n";
    ////    mat.nonzeros = cntr;
    //    mat.realloc(); // we can shrink vectors here

    //////////

    x.alloc<F>(mat.nrows, NV);
    b.alloc<F>(mat.nrows, NV);

    x.ext_offset = b.ext_offset = block_offset;

    XAMG::blas::set_const<F, NV>(x, 0.0, true);
    //    blas::set_rand<F, NV>(x, false);
    XAMG::blas::set_const<F, NV>(b, 1.0, true);
    //    blas::set_rand<F, NV>(b, false);
    //    XAMG::io::sync();

    //    auto x_ptr = x.get_aligned_ptr<F>();
    //    for (uint32_t i = 0; i < block_size; ++i)
    //        x_ptr[i] = block_offset+i;

    return true;
}
