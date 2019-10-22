#include <string>
#include <iostream>
#include <systemc.h>
#include "os_sc_fifo.h"

template <class T>
void
os_sc_fifo<T>::read( T& val_ )
{
    int task;
    while( sc_fifo<T>::num_available() == 0 ) {  
        task = os->pre_wait(); 
        sc_core::wait( sc_fifo<T>::m_data_written_event );
        os->post_wait(task);
    }
    sc_fifo<T>::m_num_read ++;
    sc_fifo<T>::buf_read( val_ );
    sc_fifo<T>::request_update();
}

template <class T>
void
accel_to_os_fifo<T>::read( T& val_ )
{
    int task;
    while( sc_fifo<T>::num_available() == 0 ) {  
        task = os->pre_wait(); 
        sc_core::wait( sc_fifo<T>::m_data_written_event );
        os->post_wait(task); 
    }
    sc_fifo<T>::m_num_read ++;
    sc_fifo<T>::buf_read( val_ );
    sc_fifo<T>::request_update();
}


template <class T>
void
os_sc_fifo<T>::write( const T& val_ )
{
    int task;
    while( sc_fifo<T>::num_free() == 0 ) {
        task = os->pre_wait();
        sc_core::wait( sc_fifo<T>::m_data_read_event );
        os->post_wait(task);
    }
    sc_fifo<T>::m_num_written ++;
    sc_fifo<T>::buf_write( val_ );
    sc_fifo<T>::request_update();
}

template <class T>
void
os_to_accel_fifo<T>::write( const T& val_ )
{
    int task;
    while( sc_fifo<T>::num_free() == 0 ) {
        task = os->pre_wait();
        sc_core::wait( sc_fifo<T>::m_data_read_event );
        os->post_wait(task);
    }
    sc_fifo<T>::m_num_written ++;
    sc_fifo<T>::buf_write( val_ );
    sc_fifo<T>::request_update();
}

