// Header file for os_channel
//

#ifndef OS_CHANNEL_H
#define OS_CHANNEL_H

#include "os_api.h"
#include <systemc.h>
#include <vector>
#include <queue>

class os_channel : public sc_channel,
                   public os_api
{
  
    // Nothing else can be public except what's defined in os_api.h!
    // And constructor/destructor obviously
    public:

    os_channel(int);
    os_channel();
    ~os_channel();

    int	pre_wait();
    void post_wait(int);
    void time_wait(int); // time in ms
    void task_terminate();
    void reg_task(const char*);
 
    private:

    int nextId = 1; // This is used to keep track of the last task ID assigned.
                    // It is only used to assign unique IDs to each task.

    int current; // Currently-sceduled task. If 0, nothing is currently running.
                 // Tasks cannot have ID = 0
    std::queue<int> readyQueue; // Queue of tasks that are ready to run

    std::vector<sc_event>    taskEvents; // Vector of events for each task
    std::vector<std::string> taskNames;  // Vector of names, for debugging purposes only.

    sc_event&   getTaskEvent(int taskId); // Returns the event associated with a task id
    std::string& getTaskName(int taskId); // Returns the name associated with a task id

    int  schedule();
    void dispatch();
    void yield();     

};

#endif
