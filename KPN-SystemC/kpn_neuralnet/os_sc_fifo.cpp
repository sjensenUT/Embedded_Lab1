#include <string>
#include <iostream>
#include <systemc.h>
#include "../kahn_process.h"
#include "os_channel.h"
#include "os_sc_fifo.h"



/*template<class T>
os_sc_fifo<T>::os_sc_fifo(os_channel *_os, int _size):
    sc_fifo<T>(_size),
    os(_os)
{
}*/

template <class T>
inline
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
inline
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


// maybe necessary for the operator
/*
operator T ()
     { return read(); }


sc_fifo<T>& operator = ( const T& a )
        { write( a ); return *this; }


template <class T>
inline
::std::ostream&
operator << ( ::std::ostream& os, const sc_fifo<T>& a )
{
    a.print( os );
    return os;
}
*/

