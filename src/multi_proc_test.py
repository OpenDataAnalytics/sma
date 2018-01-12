# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 16:29:49 2013

@author: Feng Chen
"""
import os
import multiprocessing
import time

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                print '%s: Exiting' % proc_name
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, item):
        self.item = item
    def __call__(self):
        # this is the place to do your work 
        time.sleep(0.1) # pretend to take some time to do our work
        return self.item * self.item
    def __str__(self):
        return '%s' % (self.item)


if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()

    # Start consumers
    num_consumers = 5 # We only use 5 cores. 
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    
    data = range(100)
    num_jobs = len(data)
    # Enqueue jobs
    for item in data:
        tasks.put(Task(item))
            
    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)
    
    fin = 0
    # Start printing results
    while num_jobs:
        result = results.get()
        fin = fin + result
        num_jobs -= 1
    
    print 'Result: {0}'.format(fin)

