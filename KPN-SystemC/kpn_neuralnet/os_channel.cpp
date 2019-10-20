// This is os_channel.cpp
// It is the cpp file for the os_channel class.
// I should stop making obvious redundant comments.
// TODO: Stop making comments that are redundant and obvious.

#include "os_channel.h"
#include <systemc.h>
#include <vector>
#include <queue>


os_channel::os_channel(sc_module_name name, int maxTasks)
:   sc_channel(name)
{
  cout << "in os_channel constructor" << endl; 
  this->current = 0;
  this->taskEvents = std::vector<sc_event>(maxTasks);
  this->taskNames  = std::vector<std::string>(maxTasks);
  std::cout << "[OS] Creating OS channel" << std::endl;
  std::cout << "[OS] taskEvents size = " << taskEvents.size() << std::endl;
}

os_channel::os_channel(){}


sc_event& os_channel::getTaskEvent(int taskId)
{
    if (taskId > (int) this->taskEvents.size())
    {
        std::cerr << "[OS] ERROR: Invalid task ID: " << taskId << std::endl;
    }
    return this->taskEvents[taskId-1];
}


std::string& os_channel::getTaskName(int taskId)
{ 
    if (taskId > (int) this->taskNames.size())
    {
        std::cerr << "[OS] ERROR: Invalid task ID: " << taskId << std::endl;
    }
    return this->taskNames[taskId-1];
}

double nowMs()
{
    return sc_time_stamp().to_seconds()*1000;
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
        printf("[OS] Dispatch: Scheduling task %d (%s) at time %f ms\n",
               current, getTaskName(current).c_str(), nowMs());
        // If we found a task to schedule, notify its event so that it starts running.
        // If there was nothing to schedule, schedule() returns 0.
        this->getTaskEvent(current).notify(SC_ZERO_TIME);
    } else {
        printf("[OS] Dispatch: No tasks were in the ready queue at time %f ms\n", nowMs());
    }
}


void os_channel::yield()
{
    printf("[OS] Task %d (%s) yielding at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
    int task = current;
    this->readyQueue.push(task);   // Put me back into the ready queue
    this->dispatch();              // Let someone else run now (or me if nobody else is waiting)
    wait(this->getTaskEvent(task)); // And now I will wait for my next turn.
    printf("[OS] Task %d (%s) woken up at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
}


// n is in milliseconds
void os_channel::time_wait(int n)
{
    cout << "in time_wait" << endl;
    printf("[OS] Task %d (%s) called time_wait for %d ms at time %f ms\n",
            current, getTaskName(current).c_str(), n, nowMs());
    wait(n, SC_MS);
    this->yield();
}


int os_channel::pre_wait()
{
    printf("[OS] Task %d (%s) called pre_wait at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
    int temp = current; // Remember which task called pre_wait
    this->dispatch();   // Let a new task run, as this task is about to start waiting
    return temp;        // Return the task ID to the caller so it can tell us when it's done waiting.
}


void os_channel::post_wait(int task)
{
    printf("[OS] Task %d (%s) called post_wait at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
    this->readyQueue.push(task);    // Once I'm done waiting, put me back into the ready queue
    if (!current) dispatch();       // If nobody's currently running, let me run now.
    wait(this->getTaskEvent(task)); // Now wait to be woken up.
    printf("[OS] Task %d (%s) woken up at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
}


void os_channel::task_terminate()
{
    // Let a new task run
    // I don't think there's any sort of de-allocation we need to do... not yet at least.
    printf("[OS] Terminating task %d (%s)\n", current, getTaskName(current).c_str());
    this->dispatch();
}


void os_channel::reg_task(const char* taskName)
{
    int taskId = nextId++; // nextId starts at 1, want first task to have ID 1, next 2, etc.
    printf("[OS] Registering task %s with ID %d\n", taskName, taskId);
    taskNames[taskId-1] = std::string(taskName);
    // Now we need to place the newly-registered task into the ready queue, schedule a task if
    // nothing is currently running, and wait on the event now associated with the task.
    // Currently this is exactly what post_wait does, but I don't know if that will be true 
    // for the other parts of this lab, so just duplicating the code here.
    this->readyQueue.push(taskId);
    if (!current) dispatch();
    wait(this->getTaskEvent(taskId));
    printf("[OS] Task %d (%s) scheduled for the first time at time %f ms\n",
            current, getTaskName(current).c_str(), nowMs());
}


os_channel::~os_channel() {} // Destructor
