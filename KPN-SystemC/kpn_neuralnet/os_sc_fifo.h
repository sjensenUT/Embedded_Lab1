/*
 *	kpn_fifo.cpp -- Sample code for modeling producer-consumer pair using KPN model
 *
 *	System-Level Architecture and Modeling Lab
 *	Department of Electrical and Computer Engineering
 *	The University of Texas at Austin 
 *
 * 	Author: Kamyar Mirzazad Barijough (kammirzazad@utexas.edu)
 */

//Source Code Location: 
//https://github.com/systemc/systemc-2.3/blob/master/src/sysc/communication/sc_fifo.h

#ifndef OS_SC_FIFO_H
#define OS_SC_FIFO_H

#include <string>
#include <iostream>
#include <systemc.h>
#include "../kahn_process.h"
#include "os_channel.h"

namespace sc_core {

template <class T>
class	os_sc_fifo : public sc_fifo
{
	public:
    
    // should just run the sc_fifo constructors
    explicit os_sc_fifo( int size_ = 16) : sc_fifo(size_) {}
    
    explicit os_sc_fifo( const char* name_, int size_ = 16) : sc_fifo(name_,size_) {}

    void read() override;
    void write() override;     


    private: 
    sc_fifo& operator = ( const sc_fifo<T>& );
};


template <class T>
inline
void
os_sc_fifo<T>::read( T& val_ )
{
    int task;
    while( num_available() == 0 ) {  
        task = pre_wait(); 
        sc_core::wait( m_data_written_event );
        post_wait(task); 
    }
    m_num_read ++;
    buf_read( val_ );
    request_update();
}


template <class T>
inline
void
os_sc_fifo<T>::write( const T& val_ )
{
    int task; 
    while( num_free() == 0 ) {
        task = pre_wait(); 
        sc_core::wait( m_data_read_event );
        post_wait(task); 
    }
    m_num_written ++;
    buf_write( val_ );
    request_update();
}


// maybe necessary for the operator

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


} // namespace sc_core
#endif //OS_SC_FIFO_H
