#include "anns_core.h"
#include <omp.h>
#include <pthread.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

// 比较器
struct CompareDist {
    bool operator()(const pair<float, int>& a, const pair<float, int>& b) {
        return a.first < b.first; 
    }
};

// ===== pthread 版本 =====
struct PthreadArgs {
    const float* base;
    const float* query;
    int nb, nq, dim, top_k;
    int thread_id, num_threads;
};

void* pthread_worker(void* args) {
    PthreadArgs* a = (PthreadArgs*)args;
    
    // 静态划分：每个线程处理自己的query子集
    int start = a->thread_id * (a->nq / a->num_threads);
    int end   = (a->thread_id + 1) * (a->nq / a->num_threads);
    
    for (int q = start; q < end; ++q) {
        priority_queue<pair<float,int>, 
                       vector<pair<float,int>>, 
                       CompareDist> pq;
        const float* q_vec = a->query + q * a->dim;
        for (int i = 0; i < a->nb; ++i) {
            const float* b_vec = a->base + i * a->dim;
            float dist = l2_distance_manual_simd(q_vec, b_vec, a->dim);
            pq.push({dist, i});
            if ((int)pq.size() > a->top_k) pq.pop();
        }
    }
    return nullptr;
}

void test_pthread(
    const vector<float>& base, int nb,
    const vector<float>& query, int nq, 
    int dim, int top_k, int num_threads)
{
    vector<pthread_t> threads(num_threads);
    vector<PthreadArgs> args(num_threads);
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        args[t] = {base.data(), query.data(), 
                   nb, nq, dim, top_k, t, num_threads};
        pthread_create(&threads[t], nullptr, pthread_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
    
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();
    
    cout << "规模[nb=" << nb << "] | 模式: pthread+SIMD"
         << "\t| 线程: " << num_threads
         << "\t| 耗时: " << ms << " ms" << endl;
}

// 统一的测试包装函数
void test_performance(
    const vector<float>& base, int nb,
    const vector<float>& query, int nq, int dim, int top_k, 
    string mode, int threads, string schedule_type)
{
    omp_set_num_threads(threads);
    double start_time = omp_get_wtime();

    // 支持静态/动态两种调度策略
    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < nq; ++q) {
            priority_queue<pair<float, int>, vector<pair<float, int>>, CompareDist> pq;
            const float* q_vec = query.data() + q * dim;
            for (int i = 0; i < nb; ++i) {
                const float* b_vec = base.data() + i * dim;
                float dist = 0.0f;
                if (mode == "Serial") dist = l2_distance_serial(q_vec, b_vec, dim);
                else if (mode == "AutoVec") dist = l2_distance_autovec(q_vec, b_vec, dim);
                else if (mode == "ManualSIMD") dist = l2_distance_manual_simd(q_vec, b_vec, dim);
                pq.push({dist, i});
                if (pq.size() > top_k) pq.pop();
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int q = 0; q < nq; ++q) {
            priority_queue<pair<float, int>, vector<pair<float, int>>, CompareDist> pq;
            const float* q_vec = query.data() + q * dim;
            for (int i = 0; i < nb; ++i) {
                const float* b_vec = base.data() + i * dim;
                float dist = 0.0f;
                if (mode == "Serial") dist = l2_distance_serial(q_vec, b_vec, dim);
                else if (mode == "AutoVec") dist = l2_distance_autovec(q_vec, b_vec, dim);
                else if (mode == "ManualSIMD") dist = l2_distance_manual_simd(q_vec, b_vec, dim);
                pq.push({dist, i});
                if (pq.size() > top_k) pq.pop();
            }
        }
    }

    double end_time = omp_get_wtime();
    cout << "规模[nb=" << nb << "] | 模式: " << mode 
         << "\t| 线程: " << threads 
         << "\t| 调度: " << schedule_type  // 顺便把调度方式也打印出来
         << "\t| 耗时: " << (end_time - start_time) * 1000.0 << " ms" << endl;
}

int main() {
    int dim = 96;
    int top_k = 100;
    int nq = 1000;

    cout << "===== 准备数据 =====" << endl;
    // 申请最大规模数据
    int max_nb = 100000;
    vector<float> base_data(max_nb * dim);
    vector<float> query_data(nq * dim);
    for(auto& val : base_data) val = static_cast<float>(rand()) / RAND_MAX;
    for(auto& val : query_data) val = static_cast<float>(rand()) / RAND_MAX;

    cout << "\n===== 单线程：手工SIMD vs 自动向量化 =====" << endl;
    test_performance(base_data, max_nb, query_data, nq, dim, top_k, "Serial", 1, "static");
    test_performance(base_data, max_nb, query_data, nq, dim, top_k, "AutoVec", 1, "static");
    test_performance(base_data, max_nb, query_data, nq, dim, top_k, "ManualSIMD", 1, "static");

    cout << "\n===== 多线程扩展性测试 =====" << endl;
    vector<int> dataset_sizes = {20000, 50000, 100000};
    vector<int> thread_counts = {1, 2, 4, 8};

    for (int nb : dataset_sizes) {
        cout << "\n--- 数据底库规模: " << nb << " ---" << endl;
        for (int t : thread_counts) {
            test_performance(base_data, nb, query_data, nq, dim, top_k, "ManualSIMD", t, "dynamic");
        }
    }

    cout << "\n===== 调度策略对比 (静态 vs 动态) =====" << endl;
    vector<int> sched_thread_counts = {1, 2, 4, 8};
    for (int t : sched_thread_counts) {
        test_performance(base_data, max_nb, query_data, nq, dim, top_k, 
                        "ManualSIMD", t, "static");
        test_performance(base_data, max_nb, query_data, nq, dim, top_k, 
                        "ManualSIMD", t, "dynamic");
        cout << "---" << endl;
    }

    cout << "\n===== 每个查询独占自己的队列，避免了数据写冲突，不需要加锁 =====" << endl;

    cout << "\n===== pthread vs OpenMP 性能对比 =====" << endl;
    vector<int> cmp_threads = {1, 2, 4, 8};
    for (int t : cmp_threads) {
        test_pthread(base_data, max_nb, query_data, nq, dim, top_k, t);
        test_performance(base_data, max_nb, query_data, nq, dim, top_k,
                        "ManualSIMD", t, "dynamic");
        cout << "---" << endl;
    }
    
    return 0;
}
