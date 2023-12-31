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

#if defined(__INTEL_COMPILER)
#define XAMG_VECTOR_NODEP _Pragma("ivdep")
#define XAMG_VECTOR_ALIGN _Pragma("vector aligned")
#define XAMG_NOVECTOR _Pragma("novector")
#define XAMG_RESTRICT __restrict
#elif defined(__GNUG__) // also true for Intel compiler; Intel compiler check must be ahead
#define XAMG_VECTOR_NODEP _Pragma("GCC ivdep")
#define XAMG_VECTOR_ALIGN //_Pragma("vector aligned")
#define XAMG_NOVECTOR
#define XAMG_RESTRICT __restrict__
#else
#undef XAMG_VECTOR_NODEP
#undef XAMG_VECTOR_ALIGN
#undef XAMG_NOVECTOR
#undef XAMG_RESTRICT
#endif

//#define XAMG_PERF_PRINT_DEBUG_INFO if((!id.gl_proc) && (perf.active())) { perf.print_mem_usage(); }
#define XAMG_PERF_PRINT_DEBUG_INFO

#ifdef NDEBUG
#define EXPRTOSTR(x) #x
#define ASSERT(x)                                                                                  \
    if (!(x)) {                                                                                    \
        std::cerr << "assertion failed: " << std::string(__func__) << ": " EXPRTOSTR((x))          \
                  << std::endl;                                                                    \
        abort();                                                                                   \
    }
#else
#define ASSERT(x) assert(x)
#endif
