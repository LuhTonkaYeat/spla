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

#include <random>
#include <vector>

using uint = spla::uint;

inline std::mt19937& generator() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

template<typename T>
T my_rand(T min, T max) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(generator());
}

template<>
float my_rand(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(generator());
}

template<typename T>
void fill_vector(spla::ref_ptr<spla::Vector>& vec, const std::vector<T>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
            vec->set_float(i, data[i]);
        } else if constexpr (std::is_same_v<T, int>) {
            vec->set_int(i, data[i]);
        } else if constexpr (std::is_same_v<T, uint>) {
            vec->set_uint(i, data[i]);
        }
    }
}

TEST(opencl_intersect, simple_case_float) {
    std::vector<uint>  a_keys = {0, 1, 4, 7};
    std::vector<float> a_vals = {10, 10, 5, 30};
    std::vector<uint>  b_keys = {2, 4, 5, 7, 8};
    std::vector<float> b_vals = {20, 2, 2, 2, 30};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 2);
    EXPECT_EQ(count_vals->as_int(), 2);

    uint  key;
    float val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 4);
    r_vals->get_float(0, val);
    EXPECT_FLOAT_EQ(val, 7);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 7);
    r_vals->get_float(1, val);
    EXPECT_FLOAT_EQ(val, 32);
}

TEST(opencl_intersect, simple_case_int) {
    std::vector<uint> a_keys = {0, 1, 4, 7};
    std::vector<int>  a_vals = {10, 10, 5, 30};
    std::vector<uint> b_keys = {2, 4, 5, 7, 8};
    std::vector<int>  b_vals = {20, 2, 2, 2, 30};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::INT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::INT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::INT);

    auto op = spla::PLUS_INT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 2);
    EXPECT_EQ(count_vals->as_int(), 2);

    uint key;
    int  val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 4);
    r_vals->get_int(0, val);
    EXPECT_EQ(val, 7);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 7);
    r_vals->get_int(1, val);
    EXPECT_EQ(val, 32);
}

TEST(opencl_intersect, no_keys_match_float) {
    std::vector<uint>  a_keys = {1, 3, 5, 7};
    std::vector<float> a_vals = {10, 30, 50, 70};
    std::vector<uint>  b_keys = {2, 4, 6, 8};
    std::vector<float> b_vals = {20, 40, 60, 80};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(a_keys.size(), spla::UINT);
    auto r_vals = spla::Vector::make(a_keys.size(), spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 0);
    EXPECT_EQ(count_vals->as_int(), 0);
}

TEST(opencl_intersect, all_keys_match_float) {
    std::vector<uint>  a_keys = {1, 2, 3, 4};
    std::vector<float> a_vals = {10, 20, 30, 40};
    std::vector<uint>  b_keys = {1, 2, 3, 4};
    std::vector<float> b_vals = {1, 2, 3, 4};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(a_keys.size(), spla::UINT);
    auto r_vals = spla::Vector::make(a_keys.size(), spla::FLOAT);

    auto op = spla::MULT_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 4);
    EXPECT_EQ(count_vals->as_int(), 4);

    uint  key;
    float val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 1);
    r_vals->get_float(0, val);
    EXPECT_FLOAT_EQ(val, 10);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 2);
    r_vals->get_float(1, val);
    EXPECT_FLOAT_EQ(val, 40);

    r_keys->get_uint(2, key);
    EXPECT_EQ(key, 3);
    r_vals->get_float(2, val);
    EXPECT_FLOAT_EQ(val, 90);

    r_keys->get_uint(3, key);
    EXPECT_EQ(key, 4);
    r_vals->get_float(3, val);
    EXPECT_FLOAT_EQ(val, 160);
}

TEST(opencl_intersect, random_large_float) {
    const uint N         = 10000;
    const uint KEY_RANGE = 20000;

    std::vector<uint>  a_keys(N);
    std::vector<float> a_vals(N);
    std::vector<uint>  b_keys(N);
    std::vector<float> b_vals(N);

    for (uint i = 0; i < N; ++i) {
        uint key;
        do {
            key = my_rand(0u, KEY_RANGE);
        } while (std::find(a_keys.begin(), a_keys.end(), key) != a_keys.end());
        a_keys[i] = key;
        a_vals[i] = my_rand(0.0f, 1000.0f);
    }

    for (uint i = 0; i < N; ++i) {
        uint key;
        do {
            key = my_rand(0u, KEY_RANGE);
        } while (std::find(b_keys.begin(), b_keys.end(), key) != b_keys.end());
        b_keys[i] = key;
        b_vals[i] = my_rand(0.0f, 1000.0f);
    }

    std::sort(a_keys.begin(), a_keys.end());
    std::sort(b_keys.begin(), b_keys.end());

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), count_vals->as_int());
}

TEST(opencl_intersect, different_sizes_float) {
    std::vector<uint>  a_keys = {1, 10, 100};
    std::vector<float> a_vals = {100, 200, 300};
    std::vector<uint>  b_keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100};
    std::vector<float> b_vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(a_keys.size(), spla::UINT);
    auto r_vals = spla::Vector::make(a_keys.size(), spla::FLOAT);

    auto op = spla::PLUS_FLOAT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 3);
    EXPECT_EQ(count_vals->as_int(), 3);

    uint  key;
    float val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 1);
    r_vals->get_float(0, val);
    EXPECT_FLOAT_EQ(val, 101);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 10);
    r_vals->get_float(1, val);
    EXPECT_FLOAT_EQ(val, 210);
}

TEST(opencl_intersect, subtraction_float) {
    std::vector<uint>  a_keys = {0, 1, 4, 7};
    std::vector<float> a_vals = {10, 10, 5, 30};
    std::vector<uint>  b_keys = {2, 4, 5, 7, 8};
    std::vector<float> b_vals = {20, 2, 2, 2, 30};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::FLOAT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::FLOAT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::FLOAT);

    auto op = spla::OpBinary::make_float("sub",
                                         "(float a, float b) { return a - b; }",
                                         [](float a, float b) { return a - b; });

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 2);
    EXPECT_EQ(count_vals->as_int(), 2);

    uint  key;
    float val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 4);
    r_vals->get_float(0, val);
    EXPECT_FLOAT_EQ(val, 3);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 7);
    r_vals->get_float(1, val);
    EXPECT_FLOAT_EQ(val, 28);
}

TEST(opencl_intersect, uint_type) {
    std::vector<uint> a_keys = {0, 1, 4, 7};
    std::vector<uint> a_vals = {10, 10, 5, 30};
    std::vector<uint> b_keys = {2, 4, 5, 7, 8};
    std::vector<uint> b_vals = {20, 2, 2, 2, 30};

    auto a_keys_vec = spla::Vector::make(a_keys.size(), spla::UINT);
    auto a_vals_vec = spla::Vector::make(a_vals.size(), spla::UINT);
    auto b_keys_vec = spla::Vector::make(b_keys.size(), spla::UINT);
    auto b_vals_vec = spla::Vector::make(b_vals.size(), spla::UINT);

    fill_vector(a_keys_vec, a_keys);
    fill_vector(a_vals_vec, a_vals);
    fill_vector(b_keys_vec, b_keys);
    fill_vector(b_vals_vec, b_vals);

    auto r_keys = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);
    auto r_vals = spla::Vector::make(std::min(a_keys.size(), b_keys.size()), spla::UINT);

    auto op = spla::PLUS_UINT.template cast_safe<spla::OpBinary>();

    auto status = spla::exec_intersect(a_keys_vec, a_vals_vec, b_keys_vec, b_vals_vec, r_keys, r_vals, op);
    ASSERT_EQ(status, spla::Status::Ok);

    auto count_keys = spla::Scalar::make_int(0);
    auto count_vals = spla::Scalar::make_int(0);
    spla::exec_v_count_mf(count_keys, r_keys);
    spla::exec_v_count_mf(count_vals, r_vals);
    EXPECT_EQ(count_keys->as_int(), 2);
    EXPECT_EQ(count_vals->as_int(), 2);

    uint key;
    uint val;
    r_keys->get_uint(0, key);
    EXPECT_EQ(key, 4);
    r_vals->get_uint(0, val);
    EXPECT_EQ(val, 7);

    r_keys->get_uint(1, key);
    EXPECT_EQ(key, 7);
    r_vals->get_uint(1, val);
    EXPECT_EQ(val, 32);
}

SPLA_GTEST_MAIN