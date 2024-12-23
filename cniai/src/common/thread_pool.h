#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cniai::thread_pool {

class ThreadPool {

public:
    explicit ThreadPool(size_t);
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(size_t, Args...)>::type> {

        using return_type = typename std::result_of<F(size_t, Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type(size_t)>>(
            std::bind(std::forward<F>(f), std::placeholders::_1,
                      std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](int tid) { (*task)(tid); });
        }
        condition.notify_one();
        return res;
    }

    size_t size() { return workers.size(); }

    void wait() {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        completed.wait(lock, [this] {
            return this->in_flight == 0 && this->tasks.empty();
        });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void(size_t)>> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable completed;
    int in_flight;
    bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false), in_flight(0), workers(threads) {
    if (threads <= 0)
        return;
    for (size_t i = 0; i < threads; ++i)
        workers[i] = std::thread([this, i] {
            for (;;) {
                std::function<void(size_t)> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                    in_flight++;
                }

                task(i);
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                in_flight--;
                if ((this->in_flight == 0) && this->tasks.empty())
                    completed.notify_one();
            }
        });
}

} // namespace cniai::thread_pool
