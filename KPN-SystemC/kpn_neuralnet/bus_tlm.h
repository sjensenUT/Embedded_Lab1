#ifndef BUS_TLM_H
#define BUS_TLM_H

#include <systemc.h>

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
    virtual void    masterWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d) = 0;
};

class   ISlaveTLM : virtual public sc_interface
{
    public:

    virtual void    slaveRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d) = 0;
    virtual void    slaveWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d) = 0;
};


class bus_tlm: public sc_channel, public IMasterTLM, public ISlaveTLM
{

    public:
        //sc_inout< sc_bv<DATA_WIDTH> > master_data;
        //sc_inout< sc_bv<DATA_WIDTH> > slave_data;
        sc_bv<DATA_WIDTH> data;
        bool valid = false;
        bus_tlm(sc_module_name name)  : sc_channel(name) {}
        void masterRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d){
            cout << "in masterRead" << endl;   
        }

        void masterWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d){
            cout << "in masterWrite" << endl;
        }

        void slaveRead (const sc_bv<ADDR_WIDTH>& a, sc_bv<DATA_WIDTH>& d){
            cout << "in slaveRead" << endl;
        }

        void slaveWrite(const sc_bv<ADDR_WIDTH>& a, const sc_bv<DATA_WIDTH>& d){
            cout << "in slaveWrite" << endl;
        }


};

#endif
