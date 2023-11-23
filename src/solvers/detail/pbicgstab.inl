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
struct PBiCGStab : public base_solver<F, NV> {
    const uint16_t nvecs = 8;
    const uint16_t comm_size = 3;
    DECLARE_INHERITED_FROM_BASESOLVER(PBiCGStab)
};

template <typename F, uint16_t NV>
void PBiCGStab<F, NV>::matrix_info() {
    if (precond == nullptr) {
        A.info.print("A");
    } else {
        precond->matrix_info();
    }
}

template <typename F, uint16_t NV>
void PBiCGStab<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;

    auto max_iters = param_list.get_int("max_iters");

    // inverted convergence flag; used to switch off updates to converged RHSs
    vector::vector iconv(conv);

    const vector::vector &a1 = blas::ConstVectorsCache<F>::get_ones_vec(NV);
    const vector::vector &a_1 = blas::ConstVectorsCache<F>::get_minus_ones_vec(NV);

    //////////

    vector::vector rho, res;
    rho.alloc<F>(1, NV);
    res.alloc<F>(1, NV);

    vector::vector scal1, scal2, scal3, scal4, alpha, beta, omega;
    scal1.alloc<F>(1, NV);
    scal2.alloc<F>(1, NV);
    scal3.alloc<F>(1, NV);
    scal4.alloc<F>(1, NV);
    alpha.alloc<F>(1, NV);
    beta.alloc<F>(1, NV);
    omega.alloc<F>(1, NV);

    vector::vector alpha_conv, omega_conv;
    alpha_conv.alloc<F>(1, NV);
    omega_conv.alloc<F>(1, NV);

    vector::vector &r0 = buffer[0];
    vector::vector &r = buffer[1];
    vector::vector &p = buffer[2];
    vector::vector &v = buffer[3];
    vector::vector &s = buffer[4];
    vector::vector &t = buffer[5];
    vector::vector &p_ = buffer[6];
    vector::vector &s_ = buffer[7];
    assert(nvecs == 8);
    base::set_buffers_zero();

    ///////////////////
    stats.init(r0, iconv);

    //    1. r0 = b - Ax0
    //    2. \rho_0 = (r0, r0)
    if (stats.is_converged_initial(r0, rho)) {
        return;
    }

    blas::copy<F, NV>(r0, r);

    //    3. p = r0
    blas::copy<F, NV>(r0, p);

    ////////////////////

    perf.stop();

    for (bool last_iter = false; !last_iter;) {
        auto it = stats.increment_iters_counter();
        if (it >= max_iters)
            last_iter = true;
        if (it == 2)
            perf.start();

        //        5*. p_ = M^{-1}*p
        blas::set_const<F, NV>(p_, 0.0);
        precond->solve(p_, p);

        //        5. v = A*p_
        blas2::Ax_y<F, NV>(A, p_, v, NV);

        //        6. \alpha = rho0 / (v, r0)
        blas::dot<F, NV>(v, r0, scal1);

        allreduce_buffer.push_vector(scal1);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(scal1);

        blas::xdivy_z<F, NV>(rho, scal1, alpha);

        //        7. s = r - \alpha v
        blas::scal<F, NV>(a_1, alpha);
        blas::axpby_z<F, NV>(a1, r, alpha, v, s);
        blas::scal<F, NV>(a_1, alpha);

        //        8*. s_ = M^{-1}*s
        blas::set_const<F, NV>(s_, 0.0);
        precond->solve(s_, s);

        //        8. t = A*s
        blas2::Ax_y<F, NV>(A, s_, t, NV);

        //        9. \omega = (t, s) / (t, t)
        blas::dot<F, NV>(t, s, scal2);
        blas::dot<F, NV>(t, t, scal3);
        blas::dot<F, NV>(s, s, scal4);
        allreduce_buffer.push_vector(scal2);
        allreduce_buffer.push_vector(scal3);
        allreduce_buffer.push_vector(scal4);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(scal4);
        allreduce_buffer.pull_vector(scal3);
        allreduce_buffer.pull_vector(scal2);

        blas::xdivy_z<F, NV>(scal2, scal3, omega);

        //        10. x = x + \alpha p_ + \omega s_
        //        blas::axpbypcz<F, NV>(alpha, p_, omega, s_, a1, x);
        blas::ax_y<F, NV>(iconv, alpha, alpha_conv);
        blas::ax_y<F, NV>(iconv, omega, omega_conv);
        blas::axpbypcz<F, NV>(alpha_conv, p_, omega_conv, s_, a1, x);

        //        12. Convergence check
        blas::ax_y<F, NV>(omega, scal2, res);
        blas::axpby<F, NV>(a1, scal4, a_1, res);

        if (stats.is_converged_simple(res)) {
            return;
        }

        //        11. r = s - \omega t
        blas::scal<F, NV>(a_1, omega);
        blas::axpby_z<F, NV>(a1, s, omega, t, r);
        blas::scal<F, NV>(a_1, omega);

        //        15. \rho_n = (r, r0)
        blas::dot<F, NV>(r, r0, scal1);
        allreduce_buffer.push_vector(scal1);
        allreduce_buffer.init();
        allreduce_buffer.process_sync_action();
        allreduce_buffer.wait();
        allreduce_buffer.pull_vector(scal1);

        //        16. \beta = rho_{j+1} * alpha / (rho_{j} * omega)

        blas::xdivy_z<F, NV>(scal1, rho, scal2);
        blas::xdivy_z<F, NV>(alpha, omega, scal3);
        blas::ax_y<F, NV>(scal2, scal3, beta);

        blas::copy<F, NV>(scal1, rho);

        //        17. p = r + beta p - beta*omega v
        //      order of computations can affect the convergence rate!!!
        blas::ax_y<F, NV>(omega, beta, scal1);
        blas::scal<F, NV>(a_1, scal1);

        blas::axpbypcz<F, NV>(a1, r, scal1, v, beta, p);

        perf.stop();
    }

    stats.print_residuals_footer();
}

} // namespace solver
} // namespace XAMG
