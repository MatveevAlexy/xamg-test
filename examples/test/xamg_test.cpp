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

#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include <argsparser.h>

#include <xamg/xamg_headers.h>
#include <xamg/xamg_types.h>

#include <xamg/init.h>

#include <xamg/blas/blas.h>
#include <xamg/blas2/blas2.h>
#include <xamg/solvers/solver.h>

#include <xamg/sys/dbg_helper.h>

#include "cmdline.h"

#include <cstdio>

///////////////////

extern ID id;


///////////////////

#ifndef XAMG_NV
#define XAMG_NV 16
#endif

#ifndef FP_TYPE
#define FP_TYPE float64_t
#endif

#ifdef ITAC_TRACE
#include <VT.h>
#endif

const uint16_t NV = XAMG_NV;

#include "../common/system/system.h"
#include "tst_output.h"
#include "tst_blas.h"
#include "tst_spmv.h"
#include "tst_solver.h"
#include "tst_hypre.h"

#include <cmath>
#include "../../src/comm/mpi_wrapper.h"

#include <string>
#include "../cpp/ex_solver.h"

#include "generate.hpp"



# define M_PI           3.14159265358979323846



int main(int argc, char *argv[]) {
    std::vector<std::string> solver_roles = {"meta_solver",  "solver",        "preconditioner",
                                             "pre_smoother", "post_smoother", "coarse_grid_solver"};
    args_parser parser(argc, argv, "-", ' ');
    dbg_helper dbg;
    dbg.add_parser_options(parser);
    execution_mode_t execution_mode;
    XAMG::params::global_param_list params;
    auto res = parse_cmdline(parser, solver_roles, execution_mode, params);
    switch (res) {
    case PARSE_OK:
        break;
    case PARSE_FATAL_FAILURE:
        return 1;
    case PARSE_HELP_PRINTED:
        return 0;
    }
    sleep(parser.get<int>("sleep"));
    params.set_defaults();
    dbg.get_parser_options();
    dbg.start(argv[0]);
    std::vector<std::string> logfile;
    parser.get<std::string>("logfile", logfile);
    if (logfile.size() == 1)
        logfile.push_back("");
    XAMG::init(argc, argv, parser.get<std::string>("node_config"), logfile[0], logfile[1]);
    dbg.init_output(id.gl_proc);
    tst_store_output tst_output;
    tst_output.init();
    
#ifdef ITAC_TRACE
    VT_traceoff();
#endif

    XAMG::matrix::matrix m(XAMG::mem::DISTRIBUTED);
    XAMG::vector::vector x(XAMG::mem::DISTRIBUTED), b(XAMG::mem::DISTRIBUTED);
    uint64_t n = parser.get<int>("test_iters");

    
    if (execution_mode != blas_test) {
        my_system<FP_TYPE>(parser, m, x, b, n, parser.get<bool>("graph_reordering"),
                                        parser.get<bool>("save_pattern")); 
        MPI_Barrier(MPI_COMM_WORLD);
                                        
    }

    if (execution_mode == solver_test) { // XAMG solver
        uint64_t niters = 1;
        //ex_solver_test<FP_TYPE>(m, x, b, params);
        tst_solver_test<FP_TYPE>(m, x, b, params, niters, tst_output);
        int nrows = (n + 1) * (n + 1);
        double *ans = new double[nrows];
        uint64_t block_size = nrows / id.gl_nprocs;
        uint64_t block_offset = block_size * id.gl_proc;
        if (id.gl_proc == id.gl_nprocs - 1)
            block_size = nrows - block_offset;
        double *x_d = x.get_aligned_ptr<double>();

        if (id.gl_proc) {
            MPI_Send(x_d, block_size, MPI_DOUBLE, 0, id.gl_proc, MPI_COMM_WORLD);
        }
        if (!id.gl_proc) {
            for (size_t i = 0; i < block_size; i++) {
                ans[i] = x_d[i];
            }
            for (int i = 1; i < id.gl_nprocs; i++) {
                if (i == id.gl_nprocs - 1) {
                    MPI_Recv(&ans[block_size * i], nrows - block_size * (id.gl_nprocs - 1), MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(&ans[block_size * i], block_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            
            double dif = 0, s_dif = 0;
            double avg_ans = 0, avg_el = 0;
            double h = M_PI / n / 2;

            for (size_t i = 0; i <= n; i++) {
                for (size_t j = 0; j <= n; j++) {
                    double el = cos(h * i) * sin(h * j);
                    dif = ans[i * (n + 1) + j] - el;
                    s_dif += dif * dif;
                    avg_ans += std::abs(ans[i * (n + 1) + j]);
                    avg_el += std::abs(el);
                }
            }
            std::cout << "Diff is " << sqrt(s_dif) << std::endl;
            std::cout << "Average diff is " << sqrt(s_dif) / sqrt(nrows) << std::endl;
            std::cout << "Avg ans & el: " << avg_ans / nrows << ' ' << avg_el / nrows << std::endl;
        }
    }


    XAMG::finalize();
    if (id.master_process()) {
        auto result_filename = parser.get<std::string>("result");
        if (parser.get<std::string>("load") != "") {
            tst_output.store_item("info", "config", parser.get<std::string>("load"));
        }
        tst_output.dump(result_filename);
    }
}
