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

template <typename F>
void tst_spmv_test(XAMG::matrix::matrix &m, const XAMG::vector::vector &x, XAMG::vector::vector &y,
                   uint64_t niters, tst_store_output &tst_output) {

    double dt = 0.0;
    constexpr uint64_t nwarmups = 3;
    niters = niters + nwarmups;

    // Sequence of: 0, 1, 2...
    //std::vector<F> seq;
    //for (size_t m = 0; m < NV; ++m)
    //    seq.push_back(m);
    XAMG::vector::vector csum;
    csum.alloc<uint64_t>(1, NV);
    std::vector<std::vector<uint64_t>> y_csum_history;
    y_csum_history.resize(NV);

    XAMG::mpi::barrier();
    for (uint16_t it = 0; it < niters; ++it) {
        //XAMG::blas::set_const<F, NV>(y, 0.0, true);
        // fill in vector 'y' with pre-defined randoms
        XAMG::blas::set_rand<F, NV>(y, false);

        // do set of 10 spmv-operations
        XAMG::mpi::barrier();
        double t1 = XAMG::sys::timer();
        for (uint16_t i = 0; i < 10; ++i) {
            XAMG::blas2::Axpy<F, NV>(m, x, y, NV);
        }
        XAMG::mpi::barrier();
        double t2 = XAMG::sys::timer();
        if (it >= nwarmups) {
            dt += (t2 - t1) / 10.0;
        }

        for (uint16_t nv = 0; nv < NV; ++nv) {
            XAMG::blas::bxor_global<F, NV>(y, csum);
            y_csum_history[nv].push_back(*(csum.get_aligned_ptr<uint64_t>() + nv));
        }
    }

    XAMG::out << XAMG::SUMMARY << "SUMMARY:: "
              << "SpMV: \tMatrix size: " << m.info.nrows << "\tMatrix nonzeros: " << m.info.nonzeros
              << std::endl;
    XAMG::out << XAMG::SUMMARY << "Nvecs: " << NV << "   nprocs: " << id.gl_nprocs
              << "   nnumas: " << id.nd_nnumas << "   ncores: " << id.nm_ncores
              << "   SpMV time: " << dt / (double)(niters - nwarmups) << " sec \tIters: " << niters
              << std::endl;

    bool compare_status = true;
    for (uint16_t nv = 0; nv < NV; ++nv) {
        auto &h = y_csum_history[nv];
        std::sort(h.begin(), h.end());
        if (h.front() != h.back()) {
            XAMG::out << XAMG::SUMMARY
                      << "ERROR: y vector checksums are different on spmv runs: " << h.front()
                      << ", " << h.back() << " (for vector #" << nv << ")" << std::endl;
            compare_status = false;
        }
    }
    if (!compare_status) {
        XAMG::out << "FATAL: Result vector checksums are not equal for identical Axpy() runs."
                  << std::endl;
        return;
    }

    XAMG::out.norm<F, NV>(y, " y = A * x ");
    tst_output.store_xamg_vector_norm<F, NV>("spmv", "Y_norm", y);

    tst_output.store_item("spmv", "iters", niters);

    tst_output.store_item("info", "nrows", m.info.nrows);
    tst_output.store_item("info", "nonzeros", m.info.nonzeros);
    tst_output.store_item("info", "NV", NV);
    tst_output.store_item("info", "nprocs", id.gl_nprocs);

    tst_output.store_item("topo", "nnumas", id.nd_nnumas);
    tst_output.store_item("topo", "ncores", id.nd_ncores);

    tst_output.store_item("timing", "spmv", dt / (double)(niters - nwarmups));
}
