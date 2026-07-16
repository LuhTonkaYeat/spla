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

#include "test_common.hpp"

#include <spla.hpp>

TEST(intersect, simple_case_float) {
    const spla::uint N_A = 4, N_B = 5;

    const spla::uint a_keys_data[N_A] = {0, 1, 4, 7};
    const float      a_vals_data[N_A] = {10, 10, 5, 30};
    const spla::uint b_keys_data[N_B] = {2, 4, 5, 7, 8};
    const float      b_vals_data[N_B] = {20, 2, 2, 2, 30};

    auto a_keys = spla::Vector::make(N_A, spla::UINT);
    auto a_vals = spla::Vector::make(N_A, spla::FLOAT);
    auto b_keys = spla::Vector::make(N_B, spla::UINT);
    auto b_vals = spla::Vector::make(N_B, spla::FLOAT);

    for (spla::uint i = 0; i < N_A; ++i) {
        a_keys->set_uint(i, a_keys_data[i]);
        a_vals->set_float(i, a_vals_data[i]);
    }
    for (spla::uint i = 0; i < N_B; ++i) {
        b_keys->set_uint(i, b_keys_data[i]);
        b_vals->set_float(i, b_vals_data[i]);
    }

    auto r_keys = spla::Vector::make(std::min(N_A, N_B), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(N_A, N_B), spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys, a_vals, b_keys, b_vals, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    const spla::uint expected_keys[2] = {4, 7};
    const float      expected_vals[2] = {7, 32};

    EXPECT_EQ(r_keys->get_n_rows(), 2);
    EXPECT_EQ(r_vals->get_n_rows(), 2);

    for (spla::uint i = 0; i < 2; ++i) {
        spla::uint key;
        r_keys->get_uint(i, key);
        EXPECT_EQ(key, expected_keys[i]);

        float val;
        r_vals->get_float(i, val);
        EXPECT_FLOAT_EQ(val, expected_vals[i]);
    }
}

TEST(intersect, empty_arrays_float) {
    auto a_keys = spla::Vector::make(0, spla::UINT);
    auto a_vals = spla::Vector::make(0, spla::FLOAT);
    auto b_keys = spla::Vector::make(3, spla::UINT);
    auto b_vals = spla::Vector::make(3, spla::FLOAT);

    b_keys->set_uint(0, 1);
    b_keys->set_uint(1, 2);
    b_keys->set_uint(2, 3);
    b_vals->set_float(0, 1);
    b_vals->set_float(1, 2);
    b_vals->set_float(2, 3);

    auto r_keys = spla::Vector::make(0, spla::UINT);
    auto r_vals = spla::Vector::make(0, spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys, a_vals, b_keys, b_vals, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    EXPECT_EQ(r_keys->get_n_rows(), 0);
    EXPECT_EQ(r_vals->get_n_rows(), 0);
}

TEST(intersect, no_keys_match_float) {
    const spla::uint N              = 4;
    const spla::uint a_keys_data[N] = {1, 3, 5, 7};
    const float      a_vals_data[N] = {10, 30, 50, 70};
    const spla::uint b_keys_data[N] = {2, 4, 6, 8};
    const float      b_vals_data[N] = {20, 40, 60, 80};

    auto a_keys = spla::Vector::make(N, spla::UINT);
    auto a_vals = spla::Vector::make(N, spla::FLOAT);
    auto b_keys = spla::Vector::make(N, spla::UINT);
    auto b_vals = spla::Vector::make(N, spla::FLOAT);

    for (spla::uint i = 0; i < N; ++i) {
        a_keys->set_uint(i, a_keys_data[i]);
        a_vals->set_float(i, a_vals_data[i]);
        b_keys->set_uint(i, b_keys_data[i]);
        b_vals->set_float(i, b_vals_data[i]);
    }

    auto r_keys = spla::Vector::make(N, spla::UINT);
    auto r_vals = spla::Vector::make(N, spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys, a_vals, b_keys, b_vals, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    EXPECT_EQ(r_keys->get_n_rows(), 0);
    EXPECT_EQ(r_vals->get_n_rows(), 0);
}

TEST(intersect, int_type) {
    const spla::uint N_A = 4, N_B = 5;
    const spla::uint a_keys_data[N_A] = {0, 1, 4, 7};
    const int        a_vals_data[N_A] = {10, 10, 5, 30};
    const spla::uint b_keys_data[N_B] = {2, 4, 5, 7, 8};
    const int        b_vals_data[N_B] = {20, 2, 2, 2, 30};

    auto a_keys = spla::Vector::make(N_A, spla::UINT);
    auto a_vals = spla::Vector::make(N_A, spla::INT);
    auto b_keys = spla::Vector::make(N_B, spla::UINT);
    auto b_vals = spla::Vector::make(N_B, spla::INT);

    for (spla::uint i = 0; i < N_A; ++i) {
        a_keys->set_uint(i, a_keys_data[i]);
        a_vals->set_int(i, a_vals_data[i]);
    }
    for (spla::uint i = 0; i < N_B; ++i) {
        b_keys->set_uint(i, b_keys_data[i]);
        b_vals->set_int(i, b_vals_data[i]);
    }

    auto r_keys = spla::Vector::make(std::min(N_A, N_B), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(N_A, N_B), spla::INT);

    auto op = spla::PLUS_INT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys, a_vals, b_keys, b_vals, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    const spla::uint expected_keys[2] = {4, 7};
    const int        expected_vals[2] = {7, 32};

    EXPECT_EQ(r_keys->get_n_rows(), 2);
    EXPECT_EQ(r_vals->get_n_rows(), 2);

    for (spla::uint i = 0; i < 2; ++i) {
        spla::uint key;
        r_keys->get_uint(i, key);
        EXPECT_EQ(key, expected_keys[i]);

        int val;
        r_vals->get_int(i, val);
        EXPECT_EQ(val, expected_vals[i]);
    }
}

SPLA_GTEST_MAIN