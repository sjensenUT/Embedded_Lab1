// Header file for os_dummy
//

#ifndef OS_DUMMY_H
#define OS_DUMMY_H

#include "os_api.h"
#include <systemc.h>

class os_dummy : public sc_channel,
                 public os_api
{
  
    // Nothing else can be public except what's defined in os_api.h!
    // And constructor/destructor obviously
    public:

    os_dummy(sc_module_name name) {}
    os_dummy() {}
    ~os_dummy() {}

    int	pre_wait() {}
    void post_wait(int) {}
    void time_wait(int) {}
    void task_terminate() {}
    void reg_task(const char*) {}
};

#endif
