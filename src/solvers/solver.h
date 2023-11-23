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

#include "xamg_headers.h"
#include "xamg_types.h"

#include "primitives/matrix/matrix.h"
#include "primitives/vector/vector.h"

#include "blas/blas.h"
#include "blas/blas_merged.h"

#include "hypre/hypre_wrapper.h"

#include "comm/shm_allreduce.h"

#include "stats.h"

namespace XAMG {
namespace solver {

class base_solver_interface;
template <typename F, uint16_t NV>
class base_solver;
template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A,
                                XAMG::vector::vector &x, const XAMG::vector::vector &y);
template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A);

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const params::global_param_list &params, const params::param_list &list,
                       matrix::matrix &A, const std::string &solver_role = "solver");

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const params::global_param_list &params, const params::param_list &list,
                       matrix::matrix &A, XAMG::vector::vector &x, const XAMG::vector::vector &y,
                       const std::string &solver_role = "solver");

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           const std::string &solver_role = "");

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           XAMG::vector::vector &x, const XAMG::vector::vector &y,
                           const std::string &solver_role = "");

struct base_solver_interface {
    params::param_list param_list;
    std::string solver_role = "solver";
    virtual void init(const XAMG::params::param_list &list,
                      const std::string &solver_role_ = "solver") = 0;
    virtual void setup(const XAMG::params::global_param_list &params) = 0;
    //    conv vector is used to control update of solution vector for the converged RHS:
    //    in case convergence for specific RHS was achieved the overall solution typically continues
    //    as usual, but the resulting solution vector X is NOT updated, thus the displayed residual
    //    may differ from the real one
    virtual void solve(XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(vector::vector &_x, const vector::vector &_b,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void solve(vector::vector &_x, const vector::vector &_b, const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) = 0;
    virtual void matrix_info() = 0;
    void renew_param_list(const params::param_list &l) { param_list.override_params(l); }
    void set_param_list(const params::param_list &l) {
        param_list.erase();
        param_list = l;
    }
    virtual void renew_params(const params::global_param_list &params, bool solver_mode = true) = 0;
    virtual ~base_solver_interface() {}
    virtual void set_top_solver_flag() = 0;
};

template <typename F, uint16_t NV>
struct base_solver : public base_solver_interface {
    using base_solver_interface::param_list;
    using base_solver_interface::solver_role;
    bool initialized = false;
    bool setup_done = false;
    bool top_solver_flag = false;
    matrix::matrix &A;
    vector::vector *x = nullptr;
    const vector::vector *b = nullptr;
    std::vector<vector::vector> buffer;
    std::shared_ptr<base_solver_interface> precond;
    solver_stats<F, NV> stats;
    uint16_t buffer_nvecs = 0;
    uint16_t allreduce_buffer_size = 0;
    comm::allreduce<F, NV> allreduce_buffer;
    base_solver(matrix::matrix &_A, vector::vector &_x, const vector::vector &_b)
        : A(_A), x(&_x), b(&_b) {
        stats.solver = this;
    }
    base_solver(matrix::matrix &_A) : A(_A) { stats.solver = this; }
    virtual ~base_solver() {}
    base_solver(const base_solver<F, NV> &that) = delete;
    base_solver &operator=(const base_solver<F, NV> &that) = delete;
    virtual void set_top_solver_flag() override {
        top_solver_flag = true;
        stats.set_top_solver_flag();
    }

  protected:
    void override_params(const params::param_list &l) {
        assert(!initialized);
        assert(!setup_done);
        param_list.override_params(l);
    }

  protected:
    virtual void init(const XAMG::params::param_list &list,
                      const std::string &solver_role_ = "solver") override {
        init_base(list);
    }

    void init_base(const XAMG::params::param_list &list) {
        assert(!setup_done);
        set_param_list(list);
        std::string method = list.get_string("method");

        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        buffer.resize(buffer_nvecs, vector::vector(mem::DISTRIBUTED));
        for (uint16_t i = 0; i < buffer_nvecs; ++i) {
            buffer[i].alloc<F>(numa_layer.diag.data->get_nrows(), NV);
            buffer[i].set_part(A.row_part);
        }
        allreduce_buffer.alloc(allreduce_buffer_size);
        stats.set_method(method);
        initialized = true;
    }

    virtual void setup(const XAMG::params::global_param_list &params) {
        assert(!setup_done);
        if (params.find("residual_control")) {
            stats.set_residual_control_list(params.get("residual_control"));
        }
        auto &numa_layer = A.data_layer.find(segment::NUMA)->second;
        stats.configure(numa_layer.diag.data->get_nrows(), A.row_part);
        setup_done = true;
    }

    void set_buffers_zero() {
        assert(buffer.size() == (size_t)buffer_nvecs);
        for (size_t i = 0; i < buffer_nvecs; ++i) {
            blas::set_const<F, NV>(buffer[i], 0.0);
        }
    }

  public:
    void get_residual(vector::vector &r) { get_residual(*this->x, *this->b, r); }

    void get_residual(const vector::vector &x, const vector::vector &b, vector::vector &r) {
        assert(initialized);
        assert(setup_done);

        // Computes r = b - A*x
        blas2::Ax_y<F, NV>(A, x, r, NV);
        blas::axpby<F, NV>(1.0, b, -1.0, r);
    }

  public:
    void assemble(const params::global_param_list &params,
                  const std::string &solver_role_ = "solver") {
        assert(!setup_done);
        assert(params.find(solver_role));
        init(params.get(solver_role_), solver_role_);
        assert(initialized);
        setup(params);
        assert(setup_done);
        if (solver_role == "solver" && params.find("preconditioner")) {
            auto &prms = params.get("preconditioner");
            auto method = prms.get_string("method");
            auto prec = make_solver<F, NV, base_solver_interface>(method, A);
            prec->init(prms, "preconditioner");
            prec->setup(params);
            precond = prec;
        }
    }

    virtual void renew_params(const params::global_param_list &params, bool solver_mode = true) {
        if (!solver_mode)
            return;
        assert(params.find(solver_role));
        param_list.override_params(params.get(solver_role));
        if (params.find("residual_control")) {
            stats.set_residual_control_list(params.get("residual_control"));
        }
        if (!precond)
            return;
        if (params.find("preconditioner")) {
            precond->renew_param_list(params.get("preconditioner"));
            precond->renew_params(params, false);
        }
    }

    virtual void solve(vector::vector &_x, const vector::vector &_b, const vector::vector &conv,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        assert(initialized);
        x = &_x;
        b = &_b;
        solve(conv, tok);
    }

    virtual void solve(vector::vector &_x, const vector::vector &_b,
                       XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        assert(initialized);
        x = &_x;
        b = &_b;
        const vector::vector &conv = blas::ConstVectorsCache<F>::get_ones_vec(NV);
        solve(conv, tok);
    }

    virtual void solve(XAMG::mpi::token &tok = XAMG::mpi::null_token) {
        const vector::vector &conv = blas::ConstVectorsCache<F>::get_ones_vec(NV);
        solve(conv, tok);
    }

    virtual void solve(const vector::vector &conv, XAMG::mpi::token &tok = XAMG::mpi::null_token) {}
    solver_stats<F, NV> &get_stats() {
        stats.update_results();
        return stats;
    }
};

#define DECLARE_INHERITED_FROM_BASESOLVER(CLASSNAME)                                               \
    using base = base_solver<F, NV>;                                                               \
    using base_iface = base_solver_interface;                                                      \
    using base::stats;                                                                             \
    using base::param_list;                                                                        \
    using base::buffer;                                                                            \
    using base::A;                                                                                 \
    using base::x;                                                                                 \
    using base::b;                                                                                 \
    using base::precond;                                                                           \
    using base::allreduce_buffer;                                                                  \
    CLASSNAME(matrix::matrix &_A, vector::vector &_x, const vector::vector &_b)                    \
        : base(_A, _x, _b) {                                                                       \
        base::buffer_nvecs = nvecs;                                                                \
        base::allreduce_buffer_size = comm_size;                                                   \
    }                                                                                              \
    CLASSNAME(matrix::matrix &_A) : base(_A) {                                                     \
        base::buffer_nvecs = nvecs;                                                                \
        base::allreduce_buffer_size = comm_size;                                                   \
    }                                                                                              \
    CLASSNAME(const CLASSNAME &that) = delete;                                                     \
    CLASSNAME &operator=(const CLASSNAME &that) = delete;                                          \
    virtual ~CLASSNAME() {}                                                                        \
    virtual void solve(const vector::vector &conv, XAMG::mpi::token &tok = XAMG::mpi::null_token)  \
        override;                                                                                  \
    virtual void matrix_info() override;

template <typename F, uint16_t NV>
struct Identity : public base_solver<F, NV> {
    const uint16_t nvecs = 0;
    const uint16_t comm_size = 0;
    DECLARE_INHERITED_FROM_BASESOLVER(Identity)
};

template <typename F, uint16_t NV>
void Identity<F, NV>::matrix_info() {
    A.info.print("A");
}

template <typename F, uint16_t NV>
void Identity<F, NV>::solve(const vector::vector &conv, XAMG::mpi::token &tok) {
    assert(buffer.size() == (size_t)nvecs);
    vector::vector &x = *this->x;
    const vector::vector &b = *this->b;
    uint16_t iters = 1;
    uint16_t progress = 0;
    bool flag = false;
    // if (tok != XAMG::mpi::null_token)
    //     XAMG::out << "Do progress!" << std::endl;
    for (int it = 0; it < iters; it++) {
        blas::copy<F, NV>(b, x);
        if ((tok != XAMG::mpi::null_token) && (!flag) && progress) {
            flag = XAMG::mpi::test(tok);
        }
    }
}

} // namespace solver
} // namespace XAMG

#include "detail/bicgstab.inl"
#include "detail/merged/merged_bicgstab.inl"
#include "detail/pbicgstab.inl"
#include "detail/merged/merged_pbicgstab.inl"
#include "detail/rbicgstab.inl"
#include "detail/merged/merged_rbicgstab.inl"
#include "detail/chebyshev.inl"
#include "detail/pcg.inl"
#include "detail/jacobi.inl"
#include "detail/hsgs.inl"
#include "detail/mg.inl"
#include "detail/direct.inl"
#include "detail/ir.inl"

#ifdef XAMG_EXPERIMENTAL_SOLVERS
#include "detail/experimental/ibicgstab.inl"
#include "detail/experimental/merged/merged_ibicgstab.inl"
#include "detail/experimental/pipebicgstab.inl"
#include "detail/experimental/merged/merged_pipebicgstab.inl"
#include "detail/experimental/ppipebicgstab.inl"
#include "detail/experimental/merged/merged_ppipebicgstab.inl"
#endif

namespace XAMG {
namespace solver {

#define SOLVER_START_IF_CHAIN                                                                      \
    if (false)                                                                                     \
        ;
#define SOLVER_END_IF_CHAIN else assert(0 && "The selected method is not implemented");

#define SOLVER_NEW_OPERATOR(SOLVER)                                                                \
    else if (method == #SOLVER) if (with_vectors) return std::make_shared<SOLVER<F, NV>>(_A, *x,   \
                                                                                         *y);      \
    else return std::make_shared<SOLVER<F, NV>>(_A);

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> internal_make_solver(const std::string &method, matrix::matrix &_A,
                                         XAMG::vector::vector *x, const XAMG::vector::vector *y) {
    bool with_vectors = true;
    if (x == nullptr || y == nullptr)
        with_vectors = false;
    // std::string method;
    // params.get_value("method", method);

    SOLVER_START_IF_CHAIN
    SOLVER_NEW_OPERATOR(Identity)
    SOLVER_NEW_OPERATOR(IterativeRefinement)
    SOLVER_NEW_OPERATOR(MultiGrid)
    SOLVER_NEW_OPERATOR(Direct)
    SOLVER_NEW_OPERATOR(Jacobi)
    SOLVER_NEW_OPERATOR(HSGS)
    SOLVER_NEW_OPERATOR(Chebyshev)
    SOLVER_NEW_OPERATOR(PCG)
    SOLVER_NEW_OPERATOR(BiCGStab)
    SOLVER_NEW_OPERATOR(PBiCGStab)
    SOLVER_NEW_OPERATOR(RBiCGStab)
#ifdef XAMG_EXPERIMENTAL_SOLVERS
    SOLVER_NEW_OPERATOR(IBiCGStab)
    SOLVER_NEW_OPERATOR(PipeBiCGStab)
    SOLVER_NEW_OPERATOR(PPipeBiCGStab)
#endif
    SOLVER_NEW_OPERATOR(MergedBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPBiCGStab)
    SOLVER_NEW_OPERATOR(MergedRBiCGStab)
#ifdef XAMG_EXPERIMENTAL_SOLVERS
    SOLVER_NEW_OPERATOR(MergedIBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPipeBiCGStab)
    SOLVER_NEW_OPERATOR(MergedPPipeBiCGStab)
#endif
    SOLVER_END_IF_CHAIN
    return nullptr;
}

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A,
                                XAMG::vector::vector &x, const XAMG::vector::vector &y) {
    return internal_make_solver<F, NV, RT>(method, _A, &x, &y);
}

template <typename F, uint16_t NV, class RT>
std::shared_ptr<RT> make_solver(const std::string &method, matrix::matrix &_A) {
    return internal_make_solver<F, NV, RT>(method, _A, nullptr, nullptr);
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const XAMG::params::global_param_list &params,
                       const XAMG::params::param_list &list, matrix::matrix &A,
                       const std::string &solver_role) {
    auto method = list.get_string("method");
    auto s = make_solver<F, NV, base_solver_interface>(method, A);
    s->init(list, solver_role);
    s->setup(params);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver_interface>
construct_basic_solver(const XAMG::params::global_param_list &params,
                       const XAMG::params::param_list &list, matrix::matrix &A,
                       XAMG::vector::vector &x, const XAMG::vector::vector &y,
                       const std::string &solver_role) {
    auto method = list.get_string("method");
    auto s = make_solver<F, NV, base_solver_interface>(method, A, x, y);
    s->init(list, solver_role);
    s->setup(params);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           const std::string &solver_role_) {
    bool top_solver = false;
    std::string solver_role = solver_role_;
    if (solver_role == "") {
        if (params.find("meta_solver")) {
            solver_role = "meta_solver";
        } else {
            solver_role = "solver";
        }
        top_solver = true;
    }
    auto list = params.get(solver_role);
    auto method = list.get_string("method");
    auto s = make_solver<F, NV, base_solver<F, NV>>(method, A);
    if (top_solver)
        s->set_top_solver_flag();
    s->assemble(params, solver_role);
    return s;
}

template <typename F, uint16_t NV>
std::shared_ptr<base_solver<F, NV>>
construct_solver_hierarchy(const params::global_param_list &params, matrix::matrix &A,
                           XAMG::vector::vector &x, const XAMG::vector::vector &y,
                           const std::string &solver_role_) {
    bool top_solver = false;
    std::string solver_role = solver_role_;
    if (solver_role == "") {
        if (params.find("meta_solver")) {
            solver_role = "meta_solver";
        } else {
            solver_role = "solver";
        }
        top_solver = true;
    }
    auto list = params.get(solver_role);
    auto method = list.get_string("method");
    auto s = make_solver<F, NV, base_solver<F, NV>>(method, A, x, y);
    if (top_solver)
        s->set_top_solver_flag();
    s->assemble(params, solver_role);
    return s;
}

#undef SOLVER_START_IF_CHAIN
#undef SOLVER_END_IF_CHAIN
#undef SOLVER_NEW_OPERATOR
#undef DECLARE_INHERITED_FROM_BASESOLVER
} // namespace solver
} // namespace XAMG
