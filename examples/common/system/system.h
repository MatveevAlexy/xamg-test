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

#include "../generator/generator.h"
#include "reorder.h"

template <typename F>
void make_vectors_triplet(uint32_t size, XAMG::vector::vector &x, XAMG::vector::vector &y,
                          XAMG::vector::vector &z) {
    assert(size > 0);
    uint64_t block_size = size / id.gl_nprocs;
    uint64_t block_offset = block_size * id.gl_proc;
    if (id.gl_proc == id.gl_nprocs - 1)
        block_size = size - block_offset;

    auto sh_x = std::make_shared<XAMG::vector::vector>();
    auto sh_y = std::make_shared<XAMG::vector::vector>();
    auto sh_z = std::make_shared<XAMG::vector::vector>();
    sh_x->alloc<F>(block_size, NV);
    sh_y->alloc<F>(block_size, NV);
    sh_z->alloc<F>(block_size, NV);

    sh_x->ext_offset = sh_y->ext_offset = sh_z->ext_offset = block_offset;

    XAMG::blas::set_rand<F, NV>(*sh_x, false);
    XAMG::blas::set_rand<F, NV>(*sh_y, false);
    XAMG::blas::set_rand<F, NV>(*sh_z, false);

    auto part = XAMG::part::make_partitioner(block_size);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_x, x);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_y, y);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_z, z);
}

template <typename F>
std::string make_system(args_parser &parser, XAMG::matrix::matrix &m, XAMG::vector::vector &x,
                        XAMG::vector::vector &b, const bool graph_reordering = false,
                        const bool save_pattern = false) {
    auto cmdline_matrix = parser.get<std::string>("matrix");
    using matrix_t = XAMG::matrix::csr_matrix<F, uint32_t, uint32_t, uint32_t, uint32_t>;
    auto sh_mat_csr = std::make_shared<matrix_t>();
    auto sh_x0 = std::make_shared<XAMG::vector::vector>();
    auto sh_b0 = std::make_shared<XAMG::vector::vector>();

    if (cmdline_matrix != "generate") {
        XAMG::io::read_system<matrix_t, NV>(*sh_mat_csr, *sh_x0, *sh_b0, cmdline_matrix);

    } else {
        std::map<std::string, std::string> cmdline_generator_params;
        parser.get("generator_params", cmdline_generator_params);
        generator_params_t generator_params;
        parse_generator_params(cmdline_generator_params, generator_params);
        cmdline_matrix = generate_system<F, uint32_t, uint32_t, uint32_t, uint32_t, NV>(
            *sh_mat_csr, *sh_x0, *sh_b0, generator_params);
    }

    //  use graph methods to reorder the matrix
    if (graph_reordering) {
        reorder_system(sh_mat_csr, sh_x0, sh_b0);
        assert(sh_b0->size == sh_mat_csr->nrows);
        XAMG::out << "Partitioning completed...\n";
    }

    auto part = XAMG::part::make_partitioner(sh_mat_csr->nrows);
    XAMG::matrix::construct_distributed<matrix_t>(part, *sh_mat_csr, m);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_x0, x);
    XAMG::vector::construct_distributed<F, NV>(part, *sh_b0, b);

    if (save_pattern) {
        std::string out_dir("topo");
        mkdir(out_dir.c_str(), 0777);
        m.plot(out_dir + "/" + "mat",
               XAMG::matrix::numa_diag | XAMG::matrix::numa_offd | XAMG::matrix::node_offd);
    }

    return cmdline_matrix;
}
