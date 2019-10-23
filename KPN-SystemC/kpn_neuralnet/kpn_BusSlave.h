

#ifndef KPN_BUSSLAVE_H
#define KPN_BUSSLAVE_H

#include <systemc.h>
#include "HWBus.h"

// If there were multiple bus slaves, each would need to be designated its own
// address range. However, for our case, there is just one, so let's just keep 
// it simple by hard-coding the address to 0.
static const int kAddress = 0;

class kpn_SlaveDriver : public sc_channel
{

    public:

        kpn_SlaveDriver(sc_module_name name) : sc_channel(name) {} 

        sc_port<IIntrSend> intr;
        sc_port<ISlaveHardwareBusLinkAccess> mac;
        
        void read ( void* data, unsigned long len )
        {
            // Send the interrupt to the host that we're ready for data
            intr->send();
            // Read the data from the MAC
            mac->SlaveRead(kAddress, data, len);
        }
  
        void write ( const void* data, unsigned long len )
        {
            // Notify the host that we are about to do a write
            intr->send();
            // Write the data to the MAC
            mac->SlaveWrite(kAddress, data, len);
        }
};

class kpn_BusSlave_ifc : virtual public sc_interface
{
    public:
    virtual void read ( void* data, unsigned long len ) = 0;
    virtual void write ( const void* data, unsigned long len) = 0;
};

class kpn_BusSlave : public kpn_BusSlave_ifc, public sc_channel
{
  
    public:
    
        kpn_BusSlave(sc_module_name name) :
             sc_channel(name),
             _slave("kpnBusSlave_slave"),
             _writeIrq("kpnBusSlave_writeIrq"),
             _readIrq("kpnBusSlave_readIrq"),
             _slaveMAC("kpnBusSlave_slaveMAC"),
             _slaveWriteDriver("kpnBusSlave_slaveWriteDriver"),
             _slaveReadDriver("kpnBusSlave_slaveReadDriver")
        {
            _slave.ready(ready);
            _slave.ack(ack);
            _slave.A(A);
            _slave.D(D);
            _writeIrq.intr(write_interrupt);
            _readIrq.intr(read_interrupt);
            _slaveMAC.protocol(_slave);
            _slaveWriteDriver.intr(_writeIrq);
            _slaveWriteDriver.mac(_slaveMAC);
            _slaveReadDriver.intr(_readIrq);
            _slaveReadDriver.mac(_slaveMAC);
        }

        sc_out<bool> write_interrupt, read_interrupt;
        
        sc_in< bool> ready;
        sc_out<bool> ack;
        
        sc_in< sc_bv<ADDR_WIDTH> > A;
        sc_inout< sc_bv<DATA_WIDTH> > D;

        void read ( void* data, unsigned long len )
        {
            this->_slaveReadDriver.read(data, len);
        }

        void write ( const void* data, unsigned long len )
        {
            this->_slaveWriteDriver.write(data, len);
        }

    private:

        SlaveHardwareBus _slave;
        SlaveHardwareSyncGenerate _writeIrq, _readIrq;
        SlaveHardwareBusLinkAccess _slaveMAC;
        kpn_SlaveDriver _slaveWriteDriver, _slaveReadDriver;

};

#endif
