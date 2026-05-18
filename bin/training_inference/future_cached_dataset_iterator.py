import multiprocessing
from concurrent.futures import ThreadPoolExecutor


class FutureCachedDatasetIterator(object):
    def __init__(self,
                 dataset,
                 cache_size=3,
                 iterate_only_once=False):
        self.dataset = dataset
        self.cache_size = min(cache_size, dataset.num_entries())
        self.future_dataset_dict = {x: None for x in range(self.cache_size)}
        self.iterate_only_once = iterate_only_once

        self.next_put_index = 0
        self.next_get_index = 0
        self.put_next_counter = 0
        self.get_next_counter = 0

        self.process_pool_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

        for _ in range(cache_size):
            self.put_next()

    def num_entries(self):
        return self.dataset.num_entries()

    def put_next(self):
        if self.iterate_only_once and self.put_next_counter >= self.num_entries():
            return None
        future_dataset_entry = self.process_pool_executor.submit(self.dataset.get_next)
        self.future_dataset_dict[self.next_put_index] = future_dataset_entry
        self.next_put_index = (self.next_put_index + 1) % self.cache_size
        self.put_next_counter += 1

    def get_next(self):
        if self.iterate_only_once and self.get_next_counter >= self.num_entries():
            return None
        cur_future = self.future_dataset_dict[self.next_get_index]
        dataset_entry = cur_future.result()
        self.reset_index(self.next_get_index)
        self.put_next()
        self.next_get_index = (self.next_get_index + 1) % self.cache_size
        self.get_next_counter += 1
        return dataset_entry

    def reset_index(self, index):
        cur_future = self.future_dataset_dict[index]
        self.future_dataset_dict[index] = None
        del cur_future

