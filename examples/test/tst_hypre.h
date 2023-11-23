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

#include <hypre/hypre_wrapper.h>

void tst_hypre_test(XAMG::matrix::matrix &m, XAMG::vector::vector &x, const XAMG::vector::vector &y,
                    const XAMG::params::global_param_list &params, uint64_t niters,
                    tst_store_output &tst_output) {
    double dt = 0.0;
    constexpr uint64_t nwarmups = 1;
    niters = niters + nwarmups;

    for (uint64_t it = 0; it < niters; ++it) {
        XAMG::blas::set_const<float64_t, NV>(x, 0, true);
        //XAMG::blas::set_rand<float64_t, NV>(x, false);
        XAMG::out.norm<float64_t, NV>(x, "x0");
        XAMG::out.norm<float64_t, NV>(y, "y0");
        if (it == 0) {
            tst_output.store_xamg_vector_norm<float64_t, NV>("hypre", "X_initial_norm", x);
        }
        if (NV == 1) {
            XAMG::mpi::barrier();
            double t1 = XAMG::sys::timer();
            XAMG::hypre::solve(m, x, y, params);
            XAMG::mpi::barrier();
            double t2 = XAMG::sys::timer();
            if (it >= nwarmups) {
                dt += (t2 - t1);
            }
            XAMG::out.norm<float64_t, NV>(x, " x ");
        } else {
            XAMG::out << XAMG::WARN << "Hypre test mode is allowed for single rhs vector only\n";
            break;
        }
    }
    tst_output.store_xamg_vector_norm<float64_t, NV>("hypre", "X_norm", x);
    tst_output.store_xamg_vector_norm<float64_t, NV>("hypre", "Y_norm", y);

    // XAMG::out.vector<float64_t>(XAMG::hypre::get_residual_norm(m, x, y), "||Y - A*X||");

    tst_output.store_item("info", "nrows", m.info.nrows);
    tst_output.store_item("info", "nonzeros", m.info.nonzeros);
    tst_output.store_item("info", "NV", NV);
    tst_output.store_item("info", "nprocs", id.gl_nprocs);

    tst_output.store_item("topo", "nnumas", id.nd_nnumas);
    tst_output.store_item("topo", "ncores", id.nd_ncores);

    tst_output.store_item("timing", "hypre", dt / (double)(niters - nwarmups));
}
