#############################################################################
## 
##  Copyright (C) 2019-2021 Boris Krasnopolsky, Alexey Medvedev
##  Contact: xamg-test@imec.msu.ru
## 
##  This file is part of the XAMG library.
## 
##  Commercial License Usage
##  Licensees holding valid commercial XAMG licenses may use this file in
##  accordance with the terms of commercial license agreement.
##  The license terms and conditions are subject to mutual agreement
##  between Licensee and XAMG library authors signed by both parties
##  in a written form.
## 
##  GNU General Public License Usage
##  Alternatively, this file may be used under the terms of the GNU
##  General Public License, either version 3 of the License, or (at your
##  option) any later version. The license is as published by the Free 
##  Software Foundation and appearing in the file LICENSE.GPL3 included in
##  the packaging of this file. Please review the following information to
##  ensure the GNU General Public License requirements will be met:
##  https://www.gnu.org/licenses/gpl-3.0.html.
## 
#############################################################################

# Makefile
#
# NOTE: Local Makefile to compile C API usage example.
# CC, MPICC, basic C compilation flags are specified in Make.inc/compilerflags.inc

TARGETS = examples
BASEPATH = ../..

include $(BASEPATH)/Makefile.base

LIBS += -lxamg_c_api

examples: example example_ir

example: example.o $(XAMG_C_API_LIB) $(XAMG_SEP_LIB)
	$(MPICC) $^ -o example $(LDFLAGS) $(LIBS)

example_ir: example_ir.o $(XAMG_C_API_LIB) $(XAMG_SEP_LIB)
	$(MPICC) $^ -o example_ir $(LDFLAGS) $(LIBS)

clean::
	rm -f example.o example_ir.o example example_ir
