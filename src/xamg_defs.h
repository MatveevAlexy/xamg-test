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

//#define XAMG_DBG_HEADER
//#undef XAMG_DBG_HEADER

//#define XAMG_EXPERIMENTAL_SOLVERS
#undef XAMG_EXPERIMENTAL_SOLVERS

#undef XAMG_USE_BASIC_INT_TYPES
#ifdef XAMG_LIMIT_TYPES_USAGE
#define XAMG_USE_BASIC_INT_TYPES
#endif

//#define XAMG_IO_DEBUG

//#define SHM_OPT3

//#define XAMG_MONITOR

#ifdef XAMG_DEBUG
#define XAMG_EXTRA_CHECKS
#endif
