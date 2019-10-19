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

//namespace sc_core {

template <class T>
class	os_sc_fifo: public sc_fifo<T>
{
	public:
    
    os_channel *os; 
    // should just run the sc_fifo constructors
    explicit os_sc_fifo(os_channel *os, int _size);
    
    //explicit os_sc_fifo( const char* _name, int _size);

    void read(T& val_) override;
    void write(const T& val_) override;     

    
    //private: 
    //sc_fifo& operator = ( const sc_fifo<T>& );
};

//} // namespace sc_core
#endif //OS_SC_FIFO_H
