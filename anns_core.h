#ifndef ANNS_CORE_H
#define ANNS_CORE_H

#include <iostream>
#include <vector>
#include <queue>
#include <numeric>


// 1. 串行版本的 L2 距离计算 (Baseline)
inline float l2_distance_serial(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


// 2. 编译器自动向量化的 L2 距离计算
// 通过 pragma 强行指定最高优化级别和开启循环树向量化
#pragma GCC push_options
#pragma GCC optimize ("O3", "tree-vectorize")
inline float l2_distance_autovec(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    // 提示编译器进行 SIMD 自动向量化归约
    #pragma omp simd reduction(+:dist)
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}
#pragma GCC pop_options



// 3. 针对 x86 平台的 AVX2 手工向量化

// 宏定义判断：如果是 x86 环境，则编译此段代码
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

inline float l2_distance_avx2(const float* a, const float* b, int dim = 96) {
    __m256 sum_vec = _mm256_setzero_ps(); 
    
    // 提示编译器展开循环，减少分支开销
    #pragma GCC unroll 4
    for (int i = 0; i < dim; i += 8) {
        // _mm256_loadu_ps 支持不对齐加载，提升泛用性
        __m256 va = _mm256_loadu_ps(a + i); 
        __m256 vb = _mm256_loadu_ps(b + i);
        
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    
    // 将 8 个 float 归约为 1 个
    __m128 vlow  = _mm256_castps256_ps128(sum_vec);
    __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
    vlow = _mm_add_ps(vlow, vhigh); 

    __m128 shuf = _mm_movehdup_ps(vlow);       
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);   
    sums        = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums); 
}
#endif


// 4. 针对 ARM 平台的 NEON 手工向量化 (附加分项)

// 宏定义判断：如果是 ARM 环境，则编译此段代码
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

inline float l2_distance_neon(const float* a, const float* b, int dim = 96) {
    // NEON 的 128-bit 寄存器，装 4 个 float
    float32x4_t sum_vec = vdupq_n_f32(0.0f); 
    
    #pragma GCC unroll 4
    for (int i = 0; i < dim; i += 4) { // NEON 每次处理 4 个 float
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vfmaq_f32(sum_vec, diff, diff); 
    }
    
    // 水平求和提取结果
    float temp[4];
    vst1q_f32(temp, sum_vec);
    return temp[0] + temp[1] + temp[2] + temp[3];
}
#endif


// 5. 跨平台通用 SIMD 接口包装器

inline float l2_distance_manual_simd(const float* a, const float* b, int dim = 96) {
#if defined(__x86_64__) || defined(__i386__)
    return l2_distance_avx2(a, b, dim);
#elif defined(__ARM_NEON) || defined(__aarch64__)
    return l2_distance_neon(a, b, dim);
#else
    // 如果既不是 x86 也不是 ARM (或者不支持SIMD)，退回到自动向量化版本
    return l2_distance_autovec(a, b, dim);
#endif
}

#endif // ANNS_CORE_H