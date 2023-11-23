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

#include "part/part.h"

double cycle(uint64_t niters, std::function<void(void)> f) {
    XAMG::mpi::barrier();
    double t1 = XAMG::sys::timer();
    for (uint64_t i = 0; i < niters; i++) {
        f();
    }
    double t2 = XAMG::sys::timer();
    XAMG::mpi::barrier();
    return (t2 - t1) / niters;
}

template <typename F>
void tst_blas_test( //const std::string &optype, uint64_t size,
    XAMG::vector::vector &x, XAMG::vector::vector &y, XAMG::vector::vector &z, args_parser &parser,
    uint64_t niters, tst_store_output &tst_output) {
    constexpr uint64_t nwarmups = 10;

    auto optype = parser.get<std::string>("matrix");
    auto str_size = parser.get<std::string>("generator_params");
    std::map<std::string, std::string> cmdline_generator_params;
    parser.get("generator_params", cmdline_generator_params);
    uint32_t size = std::stoi(cmdline_generator_params["vsz"]);
    ASSERT(size > 0);
    make_vectors_triplet<FP_TYPE>(size, x, y, z);

    XAMG::vector::vector a1, a2, a3;
    a1.alloc<F>(1, NV);
    a2.alloc<F>(1, NV);
    a3.alloc<F>(1, NV);
    XAMG::blas::set_const<F, NV>(a1, 1.0);
    XAMG::blas::set_const<F, NV>(a2, 0.99);
    XAMG::blas::set_const<F, NV>(a3, 0.01);

    //////////

    ////    warm-up
    for (uint64_t i = 0; i < nwarmups; ++i) {
        XAMG::blas::axpby<F, NV>(a1, x, a2, y);
        XAMG::blas::axpby<F, NV>(a1, x, a2, z);
        XAMG::blas::axpby<F, NV>(a2, y, a3, z);
    }

    double dt = 0;
    XAMG::vector::vector *res = nullptr;
    if (optype == "axpby") {
        dt = cycle(niters, [&](void) { XAMG::blas::axpby<F, NV>(a1, x, a2, y); });
        res = &y;
    } else if (optype == "axpbypcz") {
        dt = cycle(niters, [&](void) { XAMG::blas::axpbypcz<F, NV>(a1, x, a2, y, a3, z); });
        res = &z;
    } else {
        ASSERT(0 && "unknown optype in blas tests");
    }

    ASSERT(res != nullptr);

    XAMG::out.norm<F, NV>(*res, "RES");

    tst_output.store_xamg_vector_norm<F, NV>("blas", "RES_norm", *res);
    tst_output.store_xamg_vector_csum<F, NV>("blas", "RES_csum", *res);
    tst_output.store_item("blas", "iters", niters);
    tst_output.store_item("blas", "operation", optype);

    tst_output.store_item("info", "size", size);
    tst_output.store_item("info", "NV", NV);
    tst_output.store_item("info", "nprocs", id.gl_nprocs);

    tst_output.store_item("topo", "nnumas", id.nd_nnumas);
    tst_output.store_item("topo", "ncores", id.nd_ncores);

    tst_output.store_item("timing", "blas", dt);
}
