#ifndef OS_SC_FIFO_H
#define OS_SC_FIFO_H

#include <string>
#include <iostream>
#include <systemc.h>
#include "../kahn_process.h"
#include "os_channel.h"

template <class T>
class	os_sc_fifo: public sc_fifo<T>
{
	public:
    
    sc_port<os_channel> os; 
    explicit os_sc_fifo(int _size): sc_fifo<T>(_size){};
    
    //explicit os_sc_fifo( const char* _name, int _size);

    void read(T& val_) override;
    void write(const T& val_) override;     

    
    //private: 
    //sc_fifo& operator = ( const sc_fifo<T>& );
};

template <class T>
class	os_to_accel_fifo: public sc_fifo<T>
{
	public:
    
    sc_port<os_channel> os; 
    explicit os_to_accel_fifo(int _size): sc_fifo<T>(_size){};
    
    void write(const T& val_) override;
};

template <class T>
class	accel_to_os_fifo: public sc_fifo<T>
{
	public:
    
    sc_port<os_channel> os; 
    explicit accel_to_os_fifo(int _size): sc_fifo<T>(_size){};
    
    void read(T& val_) override;    
};

#endif //OS_SC_FIFO_H
