// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Barrier is an object that allows one or more threads to wait until
// Notify has been called a specified number of times.

#ifndef EIGEN_CXX11_THREADPOOL_BARRIER_H
#define EIGEN_CXX11_THREADPOOL_BARRIER_H

#include <unistd.h>
#include <sys/syscall.h>

#include <atomic>
#include <thread>

namespace Eigen {

class Barrier {
 public:
  Barrier(unsigned int count) : state_(count << 1), notified_(false) {
    eigen_plain_assert(((count << 1) >> 1) == count);
  }
  ~Barrier() { eigen_plain_assert((state_ >> 1) == 0); }

  void Notify() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      eigen_plain_assert(((v + 2) & ~1) != 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::unique_lock<std::mutex> l(mu_);
    eigen_plain_assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return;
    std::unique_lock<std::mutex> l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<unsigned int> state_;  // low bit is waiter flag
  bool notified_;
};

// Notification is an object that allows a user to to wait for another
// thread to signal a notification that an event has occurred.
//
// Multiple threads can wait on the same Notification object,
// but only one caller must call Notify() on the object.
struct Notification : Barrier {
  Notification() : Barrier(1){};
};

inline int gettid() {
  return syscall(__NR_gettid);
}

// Simple busy-wait spinlock (Lockable)
class BWSpinLock {
 public:
  bool try_lock_() {
    bool expected = false;
    return flag_.compare_exchange_strong(
        expected, true, std::memory_order_acquire);
  }
  void lock() {
#ifndef NDEBUG
    int this_thread_tid = gettid();
    int owner_thread_tid = owner_tid_.load();
    if (this_thread_tid == owner_thread_tid) {
      printf("\033[0;31mRecursive lock!\033[0m\n");
      eigen_plain_assert(owner_thread_tid != this_thread_tid);
    }

    uint64_t spin_count = 0;
#endif
    while (!try_lock_()) {
#ifndef NDEBUG
      if (++spin_count > 10000000) {
        printf("\033[0;31mThread %d stuck waiting for thread %d "
            "to release the lock\033[0m\n", this_thread_tid, owner_thread_tid);
        spin_count = 0;
      }
#endif
      std::this_thread::yield();
    };
#ifndef NDEBUG
    owner_tid_.store(gettid());
#endif
    eigen_plain_assert(flag_.load() == true);
  }
  void unlock() {
    eigen_plain_assert(flag_.load() == true);
#ifndef NDEBUG
    eigen_plain_assert(gettid() == owner_tid_.load());
    owner_tid_.store(0, std::memory_order_release);
#endif
    flag_.store(false, std::memory_order_release);
  }
 private:
#ifndef NDEBUG
  std::atomic<int> owner_tid_{0};
#endif
  std::atomic<bool> flag_{false};
};

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_BARRIER_H

// vim: ts=2 sw=2
