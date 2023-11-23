/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2019-2020, Boris Krasnopolsky, Alexey Medvedev
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <sstream>
#include <numa.h>
#include <map>
#include <vector>
#ifdef WITH_CUDA
#include <nvml.h>
#endif

union cpuset {
    unsigned long set[2];
    struct {
        unsigned long w1, w2;
    } words;
    cpuset() { words.w1 = words.w2 = 0; }
};

struct node_elements {
    std::vector<int> cpus;
    std::vector<int> gpus;
};

int main()
{
    std::map<int, node_elements> nodes; 
    int nnodes = numa_num_configured_nodes();
    int ncpus = numa_num_configured_cpus();
    std::cout << ">> " << "nnodes=" << nnodes << " ncpus=" << ncpus << std::endl;
    for (int i = 0; i < ncpus; i++) {
        int node = numa_node_of_cpu(i);
        nodes[node].cpus.push_back(i); 
    }
    std::vector<cpuset> cpusets;
    for (auto &n : nodes) {
        int node = n.first;
        auto cpus = n.second.cpus;
        cpuset s;
        for (auto c : cpus) {
            std::cout << ">> " << "node=" << node << " cpu=" << c << std::endl;
            if (c > 63)
                s.words.w2 |= (1 << (c - 64));
            else
                s.words.w1 |= (1 << c);
        }
        cpusets.push_back(s);
    }
#ifdef WITH_CUDA
    unsigned int devcnt = 0;
    nvmlDeviceGetCount(&devcnt);
    std::cout << ">> "  << "devcnt=" << devcnt << std::endl;
    for (int i = 0; i < devcnt; i++) {
        cpuset s;
        nvmlDevice_t id;
        nvmlDeviceGetHandleByIndex(i, &id);
        nvmlDeviceGetCpuAffinity(id, 2, s.set);
        for (auto &n : nodes) {
            int node = n.first;
            auto &gpus = n.second.gpus;
            if (cpusets[node].words.w1 == s.words.w1 && cpusets[node].words.w2 == s.words.w2) {
                std::cout << ">> "  << "device=" << i << " -> node=" << node << std::endl;
                gpus.push_back(i);
            }
        }
    }
#endif
    {
        std::ostringstream ss;
        for (auto &n : nodes) {
            int node = n.first;
            auto cpus = n.second.cpus;
            auto gpus = n.second.gpus;
            for (int c : cpus) {
                ss << c << ",";
            }
            ss.seekp(-1, std::ios_base::cur);
            ss << "@";
            for (int g : gpus) {
                ss << g << ",";
            }
            ss.seekp(-1, std::ios_base::cur);
            ss << ";";
        }
        ss.seekp(-1, std::ios_base::cur);
        ss << '\0';
        std::cout << ss.str() << std::endl;
    }
    {
        std::ostringstream ss;
        for (auto &n : nodes) {
            int node = n.first;
            auto cpus = n.second.cpus;
            auto gpus = n.second.gpus;
            cpuset cs;
            for (int c : cpus) {
                if (c > 63) {
                    cs.words.w2 |= (1 << (c - 64));
                } else {
                    cs.words.w1 |= (1 << c);
                }
            }
            ss << "0x" << std::hex;
            if (cs.words.w2)
                ss << std::hex << cs.words.w2;
            ss << cs.words.w1;
            ss << std::dec;
            ss << "@";
            for (int g : gpus) {
                ss << g << ",";
            }
            ss.seekp(-1, std::ios_base::cur);
            ss << ";";
        }
        ss.seekp(-1, std::ios_base::cur);
        ss << '\0';
        std::cout << ss.str() << std::endl;
    }
    return 0;
}
