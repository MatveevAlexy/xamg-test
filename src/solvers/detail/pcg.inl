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

namespace XAMG {
namespace solver {

template <typename F, uint16_t NV>
struct PCG : public base_solver<F, NV> {
    const uint16_t nvecs = 4;
    const uint16_t comm_size = 3;
    DECLARE_INHERITED_FROM_BASESOLVER(PCG)
};

template <typename F, uint16_t NV>
void PCG<F, NV>::matrix_info() {
    if (precond == nullptr) {
        A.info.print("A");
    } else {
        precond->matrix_info();
    }
}

template <typename F, uint16_t NV>
void PCG<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    auto max_iters = param_list.get_int("max_iters");

    // inverted convergence flag; used to switch off updates to already converged RHSs
    vector::vector iconv(conv);

    // cached constant vectors
    const vector::vector &a0 = blas::ConstVectorsCache<F>::get_zeroes_vec(NV);
    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);
    const vector::vector &a_1 = blas::ConstVectorsCache<F>::get_minus_ones_vec(NV);

    // residual which is incrementally re-calculated on every step
    vector::vector res;
    res.alloc<F>(1, NV);

    vector::vector scal1, scal2, scal3, alpha, beta;
    scal1.alloc<F>(1, NV);
    scal2.alloc<F>(1, NV);
    scal3.alloc<F>(1, NV);
    alpha.alloc<F>(1, NV);
    beta.alloc<F>(1, NV);

    vector::vector alpha_conv;
    alpha_conv.alloc<F>(1, NV);

    vector::vector &r = buffer[0];
    vector::vector &z = buffer[1];
    vector::vector &p = buffer[2];
    vector::vector &Ap = buffer[3];
    assert(nvecs == 4);
    assert(buffer.size() >= (size_t)nvecs);
    base::set_buffers_zero();

    ///////////////////

    stats.init(r, iconv);
    if (stats.is_converged_initial(r)) {
        return;
    }

    //    z0 = M^{-1} r0
    blas::set_const<F, NV>(z, 0.0);
    precond->solve(z, r);

    blas::copy<F, NV>(z, p);

    ////////////////////

    perf.stop();
    for (bool last_iter = false; !last_iter;) {
        auto it = stats.increment_iters_counter();
        if (it >= max_iters)
            last_iter = true;
        if (it == 2)
            perf.start();

        blas2::Ax_y<F, NV>(A, p, Ap, NV);

        blas::dot<F, NV>(r, z, scal1);
        blas::dot<F, NV>(p, Ap, scal2);
        allreduce_buffer.push_vector(scal1);
        allreduce_buffer.push_vector(scal2);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(scal2);
        allreduce_buffer.pull_vector(scal1);

        blas::xdivy_z<F, NV>(scal1, scal2, alpha);

        //        blas::axpby<F, NV>(alpha, p, a1, x);
        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::axpby<F, NV>(alpha_conv, p, a1, x);

        blas::scal<F, NV>(a_1, alpha);
        blas::axpby<F, NV>(alpha, Ap, a1, r);

        //    z0 = M^{-1} r0
        blas::set_const<F, NV>(z, 0.0);
        precond->solve(z, r);

        blas::dot<F, NV>(r, z, scal3);
        allreduce_buffer.push_vector(scal3);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(scal3);

        blas::xdivy_z<F, NV>(scal3, scal1, beta);

        blas::axpby<F, NV>(a1, z, beta, p);

        blas::dot<F, NV>(r, r, res);
        allreduce_buffer.push_vector(res);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(res);

        if (stats.is_converged_simple(res)) {
            return;
        }
        perf.stop();
    }

    stats.print_residuals_footer();
}

} // namespace solver
} // namespace XAMG
