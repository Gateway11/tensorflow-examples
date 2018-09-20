
//namespace {

int pthread_barrier_init(pthread_barrier_t* barrier,
                         const void* barrier_attr,
                         unsigned count) {
  barrier->count = count;
  pthread_mutex_init(&barrier->mutex, NULL);
  pthread_cond_init(&barrier->cond, NULL);
  return 0;
}

int pthread_barrier_wait(pthread_barrier_t* barrier) {
  // Lock the mutex
  pthread_mutex_lock(&barrier->mutex);
  // Decrement the count. If this is the first thread to reach 0, wake up
  // waiters, unlock the mutex, then return PTHREAD_BARRIER_SERIAL_THREAD.
  if (--barrier->count == 0) {
    // First thread to reach the barrier
    pthread_cond_broadcast(&barrier->cond);
    pthread_mutex_unlock(&barrier->mutex);
    return PTHREAD_BARRIER_SERIAL_THREAD;
  }
  // Otherwise, wait for other threads until the count reaches 0, then
  // return 0 to indicate this is not the first thread.
  do {
    pthread_cond_wait(&barrier->cond, &barrier->mutex);
  } while (barrier->count > 0);

  pthread_mutex_unlock(&barrier->mutex);
  return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier) {
  barrier->count = 0;
  pthread_cond_destroy(&barrier->cond);
  pthread_mutex_destroy(&barrier->mutex);
  return 0;
}

int pthread_yield(void) {
  sched_yield();
  return 0;
}

//}  // namespace
