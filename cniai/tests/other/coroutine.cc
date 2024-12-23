#include <coroutine>
#include <iostream>
#include <list>
#include <thread>
#include <unistd.h>
#include <vector>

class CoroutineAwaitable {
public:
    virtual bool waiting() = 0;
};

template <typename Awaitable>
struct Generator {

    struct promise_type {
        CoroutineAwaitable *coroutineExecutorAwaitable;
        std::exception_ptr mException;

        std::suspend_never initial_suspend() {
            std::cout << "initial_suspend called\n";
            return {};
        }

        std::suspend_always final_suspend() noexcept {
            std::cout << "final_suspend called\n";
            return {};
        }

        void unhandled_exception() {
            mException = std::current_exception();
            std::cout << "final_suspend called\n";
            abort();
        }

        Generator get_return_object() {
            std::cout << "get_return_object called\n";
            return Generator(std::coroutine_handle<promise_type>::from_promise(*this));
        }

        void return_void() {}

        template <typename U>
        U &&await_transform(U &&awaitable) noexcept {
            return static_cast<U &&>(awaitable);
        }
    };

    explicit Generator(std::coroutine_handle<promise_type> handle) : mHandle(handle) {}

    ~Generator() {
        std::cout << " ~Generator()" << std::endl;
        mHandle.destroy();
    }

    bool canResume() { return !mHandle.promise().coroutineExecutorAwaitable->waiting(); }

    Awaitable mAwaitable;
    std::coroutine_handle<promise_type> mHandle;
};

class Payload {
public:
    std::vector<int> _flag;
};

struct CoroutineExecutorAwaitable : public CoroutineAwaitable {
    Payload payload;

    bool await_ready() {
        return false;
        std::cout << "await_ready called\n";
        std::cout << "flag: " << payload._flag[0] << std::endl;
        if (payload._flag[0] == 1) {
            std::cout << "await_ready return true\n";
            return true;
        }
        std::cout << "await_ready return false\n";
        return false;
    }

    void await_suspend(std::coroutine_handle<Generator<int>::promise_type> handle) {
        std::cout << "await_suspend called\n";
        payload._flag.push_back(0);

        handle.promise().coroutineExecutorAwaitable = this;

        //        std::cout << "do something" << std::endl;
        //
        //        usleep(5000000);
        //
        //        std::cout << "done something" << std::endl;

        std::cout << "await_suspend called done\n";
    };

    std::string await_resume() {
        std::cout << "await_resume called\n";
        if (payload._flag[0] != 1) {
            std::cout << "Fuck u, did not finished, but returned. \n";
        }
        return std::to_string(payload._flag[0]);
    }

    bool waiting() override { return payload._flag[0] != 1; }
};

Generator<int> coroutine() {
    CoroutineExecutorAwaitable coroutineExecutorAwaitable;
    std::thread t([&]() {
        usleep(5000000);
        coroutineExecutorAwaitable.payload._flag[0] = 1;
        std::cout << "Hello from thread: " << std::this_thread::get_id() << "\n";
    });

    auto ret = co_await coroutineExecutorAwaitable;
    std::cout << "co_await ret: " << ret << std::endl;

    t.join();

    co_return;
}

int main() {
    std::list<Generator<int>> gens;
    auto _ret = coroutine();
    gens.emplace_back(_ret);
    std::cout << "main thread call resume" << std::endl;
    auto ret = *gens.begin();
    while (!ret.canResume()) {
        usleep(1000000);
        std::cout << "isCurrentAwaitDone false\n";
    }
    ret.mHandle.resume();

    std::cout << "ret done? " << ret.mHandle.done() << std::endl;
    //    ret.mHandle.resume();
    //    std::cout << "ret done? " << ret.mHandle.done() << std::endl;

    return 0;
}