#ifndef BUS_TLM_H
#define BUS_TLM_H

#include <systemc.h>
#include "os_sc_fifo.h"

#define ADDR_WIDTH  16u
#define DATA_WIDTH  32u

#if DATA_WIDTH == 32u
# define DATA_BYTES 4u
#elif DATA_WIDTH == 16u
# define DATA_BYTES 2u
#elif DATA_WIDTH == 8u
# define DATA_BYTES 1u
#else
# error "Invalid data width"
#endif


class   IMasterTLM : virtual public sc_interface
{
    public:

    virtual void    masterRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d) = 0;
    virtual void    masterWrite (const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d) = 0;
};

class   ISlaveTLM : virtual public sc_interface
{
    public:

    virtual void    slaveRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d) = 0;
    virtual void    slaveWrite (const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d) = 0;
};


class bus_tlm: public sc_channel, public IMasterTLM, public ISlaveTLM
{

    public:
        
        
        os_to_accel_fifo<sc_bv<DATA_WIDTH>> os_to_accel;
        accel_to_os_fifo<sc_bv<DATA_WIDTH>> accel_to_os;
        
        bus_tlm(sc_module_name name, os_channel *os)  
        : sc_channel(name), 
          os_to_accel(1),
          accel_to_os(1)
        {
            os_to_accel.os(*os);
            accel_to_os.os(*os); 
        }

        void masterRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d){
            //cout << "in masterRead" << endl;
            accel_to_os.read(d);
        }

        void masterWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d){
            //cout << "in masterWrite" << endl;
            os_to_accel.write(d);
        }

        void slaveRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d){
            //cout << "in slaveRead" << endl;
            os_to_accel.read(d);

        }

        void slaveWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d){
            //cout << "in slaveWrite" << endl;
            accel_to_os.write(d);
        }


};

#endif
