/**********************************************************************************/
/* This file is part of spla project                                              */
/* https://github.com/SparseLinearAlgebra/spla                                    */
/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2023 SparseLinearAlgebra                                         */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#ifndef SPLA_CPU_INTERSECT_HPP
#define SPLA_CPU_INTERSECT_HPP

#include <schedule/schedule_tasks.hpp>

#include <core/dispatcher.hpp>
#include <core/logger.hpp>
#include <core/registry.hpp>
#include <core/top.hpp>
#include <core/tscalar.hpp>
#include <core/ttype.hpp>
#include <core/tvector.hpp>

#include <cstdint>
#include <vector>

namespace spla {

    template<typename T>
    class Algo_intersect_cpu final : public RegistryAlgo {
    public:
        ~Algo_intersect_cpu() override = default;

        std::string get_name() override {
            return "intersect";
        }

        std::string get_description() override {
            return "sequential set intersection of two key-value arrays";
        }

        Status execute(const DispatchContext& ctx) override {
            auto t = ctx.task.template cast_safe<ScheduleTask_intersect>();

            ref_ptr<TVector<uint32_t>> a_keys_vec = t->a_keys.template cast_safe<TVector<uint32_t>>();
            ref_ptr<TVector<T>>        a_vals_vec = t->a_vals.template cast_safe<TVector<T>>();
            ref_ptr<TVector<uint32_t>> b_keys_vec = t->b_keys.template cast_safe<TVector<uint32_t>>();
            ref_ptr<TVector<T>>        b_vals_vec = t->b_vals.template cast_safe<TVector<T>>();

            ref_ptr<TVector<uint32_t>> r_keys_vec = t->r_keys.template cast_safe<TVector<uint32_t>>();
            ref_ptr<TVector<T>>        r_vals_vec = t->r_vals.template cast_safe<TVector<T>>();

            ref_ptr<TOpBinary<T, T, T>> op = t->op.template cast_safe<TOpBinary<T, T, T>>();

            r_keys_vec->validate_wd(FormatVector::CpuDok);
            r_vals_vec->validate_wd(FormatVector::CpuDok);
            a_keys_vec->validate_rw(FormatVector::CpuDok);
            a_vals_vec->validate_rw(FormatVector::CpuDok);
            b_keys_vec->validate_rw(FormatVector::CpuDok);
            b_vals_vec->validate_rw(FormatVector::CpuDok);

            auto*       p_r_keys = r_keys_vec->template get<CpuDokVec<uint32_t>>();
            auto*       p_r_vals = r_vals_vec->template get<CpuDokVec<T>>();
            const auto* p_a_keys = a_keys_vec->template get<CpuDokVec<uint32_t>>();
            const auto* p_a_vals = a_vals_vec->template get<CpuDokVec<T>>();
            const auto* p_b_keys = b_keys_vec->template get<CpuDokVec<uint32_t>>();
            const auto* p_b_vals = b_vals_vec->template get<CpuDokVec<T>>();

            p_r_keys->Ax.clear();
            p_r_vals->Ax.clear();
            p_r_keys->values = 0;
            p_r_vals->values = 0;

            const auto& function = op->function;

            auto it_a  = p_a_keys->Ax.begin();
            auto it_b  = p_b_keys->Ax.begin();
            auto end_a = p_a_keys->Ax.end();
            auto end_b = p_b_keys->Ax.end();

            while (it_a != end_a && it_b != end_b) {
                uint32_t key_a = it_a->first;
                uint32_t key_b = it_b->first;

                if (key_a < key_b) {
                    ++it_a;
                } else if (key_b < key_a) {
                    ++it_b;
                } else {
                    p_r_keys->Ax[key_a] = key_a;
                    p_r_vals->Ax[key_a] = function(it_a->second, it_b->second);
                    p_r_keys->values++;
                    p_r_vals->values++;
                    ++it_a;
                    ++it_b;
                }
            }

            LOG_MSG(Status::Ok, "Found " << p_r_keys->values << " matches");
            return Status::Ok;
        }
    };

}//namespace spla

#endif//SPLA_CPU_INTERSECT_HPP