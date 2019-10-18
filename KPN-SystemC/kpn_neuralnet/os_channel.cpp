// This is os_channel.cpp
// It is the cpp file for the os_channel class.
// I should stop making obvious redundant comments.
// TODO: Stop making comments that are redundant and obvious.

#include "os_channel.h"
#include <systemc.h>
#include <vector>
#include <queue>


os_channel::os_channel(int maxTasks)
{
  this->current = 0;
  this->taskEvents = std::vector<sc_event>(maxTasks);
  std::cout << "taskEvents size = " << taskEvents.size() << std::endl;
}


int os_channel::schedule()
{
    // Simply pop the value on top of the queue.
    // If the queue is empty, return 0.
    if (this->readyQueue.empty()) return 0;
    int t = this->readyQueue.front();
    this->readyQueue.pop();
    return t;
}


void os_channel::dispatch()
{
    current = this->schedule(); // Schedule a new task
    if (current)
    {
        // If we found a task to schedule, notify its event so that it starts running.
        // If there was nothing to schedule, schedule() returns 0.
        this->taskEvents[current].notify(SC_ZERO_TIME);
    }
}


void os_channel::yield()
{
    int task = current;
    this->readyQueue.push(task);  // Put me back into the ready queue
    this->dispatch();             // Let someone else run now (or me if nobody else is waiting)
    wait(this->taskEvents[task]); // And now I will wait for my next turn.
}


// n is in milliseconds
void os_channel::time_wait(int n)
{
    wait(n, SC_MS);
    this->yield();
}


int os_channel::pre_wait()
{
    int temp = current; // Remember which task called pre_wait
    this->dispatch();   // Let a new task run, as this task is about to start waiting
    return temp;        // Return the task ID to the caller so it can tell us when it's done waiting.
}


void os_channel::post_wait(int task)
{
    this->readyQueue.push(task);  // Once I'm done waiting, put me back into the ready queue
    if (!current) dispatch();     // If nobody's currently running, let me run now.
    wait(this->taskEvents[task]); // Now wait to be woken up.
}


void os_channel::task_terminate()
{
    // Let a new task run
    // I don't think there's any sort of de-allocation we need to do... not yet at least.
    printf("Terminating task with ID %d\n", current);
    this->dispatch();
}


void os_channel::reg_task(const char* taskName)
{
    int taskId = nextId++; // nextId starts at 1, want first task to have ID 1, next 2, etc.
    printf("Registering task %s with ID %d\n", taskName, taskId);
    // Now we need to place the newly-registered task into the ready queue, schedule a task if
    // nothing is currently running, and wait on the event now associated with the task.
    // Currently this is exactly what post_wait does, but I don't know if that will be true 
    // for the other parts of this lab, so just duplicating the code here.
    this->readyQueue.push(taskId);
    if (!current) dispatch();
    wait(this->taskEvents[taskId]);
}


os_channel::~os_channel() {} // Destructor
