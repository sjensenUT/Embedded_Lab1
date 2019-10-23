#ifndef KPN_BUSMASTER_H
#define KPN_BUSMASTER_H

#include <systemc.h>
#include "HWBus.h"
#include "../kahn_process.h"

// If there were multiple bus slaves, each would need to be designated its own
// address range. However, for our case, there is just one, so let's just keep 
// it simple by hard-coding the address to 0.
static const int kAddressM = 0;


class kpn_MasterInterruptThread : public kahn_process
{
    public:
    
    kpn_MasterInterruptThread(sc_module_name name, sc_event& ev) : 
        kahn_process(name),
        clearIrq(ev)
    {
    
    }

    sc_port<IIntrRecv> irq;
    sc_out<bool>       irqFlag;
    sc_event&          clearIrq;

    private:

    void init() {}

    void process()
    {
        while(true)
        {
            // Wait for the slave to assert the IRQ
            irq->receive();
            // Assert the flag
            irqFlag = true;
            // Wait for the master driver to clear the IRQ,
            // clear the flag, and repeat.
            wait(clearIrq);
            irqFlag = false;
        }
    }
};


class kpn_MasterDriver : public sc_channel
{

    public:

    kpn_MasterDriver(sc_module_name name, sc_event& ev) : 
        sc_channel(name),
        clearIrq(ev)
    {
        
    } 

    sc_port<IMasterHardwareBusLinkAccess> mac;
    sc_port<os_api> os;
    sc_in<bool>     irqFlag;
    sc_event&       clearIrq;

    void read ( void* data, unsigned long len )
    {
        // Wait for the slave to indicate it is about to start a write..
        waitForInterrupt();
        // Read the data from the MAC
        mac->MasterRead(kAddressM, data, len);
    }

    void write ( const void* data, unsigned long len )
    {
        // Wait for the slave to indicate it's ready for a write.
        waitForInterrupt();
        // Write the data to the MAC
        mac->MasterWrite(kAddressM, data, len);
    }

    private:
    
    void waitForInterrupt()
    {
        // Check the interrupt flag. If it's already asserted, then we're ready
        // to proceed with the read or write. Otherwise, wait for it to assert.
        if (!irqFlag)
        {
            int task = os->pre_wait();
            wait(irqFlag.posedge_event());
            os->post_wait(task);
        }
        // Clear the interrupt right now to make room for the next one.
        // that is, if there is a next one...
        clearIrq.notify(SC_ZERO_TIME);
    }

};

class kpn_BusMaster_ifc : virtual public sc_interface
{
    public:
    virtual void read ( void* data, unsigned long len ) = 0;
    virtual void write ( const void* data, unsigned long len) = 0;
};

class kpn_BusMaster : public kpn_BusMaster_ifc, public sc_channel
{

    public:
  
    kpn_BusMaster(sc_module_name name) :
         sc_channel(name),
         _master("kpnBusMaster_master"),
         _writeIrq("kpnBusMaster_writeIrq"),
         _readIrq("kpnBusMaster_readIrq"),
         _masterMAC("kpnBusMaster_slaveMAC"),
         _masterWriteDriver("kpnBusMaster_masterWriteDriver", _clearWriteIrq),
         _masterReadDriver("kpnBusMaster_masterReadDriver", _clearReadIrq),
         _writeIrqThread("kpnBusMaster_writeIrqThread", _clearWriteIrq),
         _readIrqThread("kpnBusMaster_readIrqThread", _clearReadIrq)
    {
        // Master bus protocol connections
        _master.ready(ready);
        _master.ack(ack);
        _master.A(A);
        _master.D(D);
        // Interrupt detection connections
        _writeIrq.intr(write_interrupt);
        _readIrq.intr(read_interrupt);
        // Master MAC connections
        _masterMAC.protocol(_master);
        // Master write driver connections
        _masterWriteDriver.mac(_masterMAC);
        _masterWriteDriver.os(os);
        _masterWriteDriver.irqFlag(_writeIrqFlag);
        //_masterWriteDriver.clearIrq(_clearWriteIrq);
        // Master read driver connections
        _masterReadDriver.mac(_masterMAC);
        _masterReadDriver.os(os);
        _masterReadDriver.irqFlag(_readIrqFlag);
        //_masterReadDriver.clearIrq(_clearReadIrq);
        // Master write interrupt thread connections
        _writeIrqThread.irq(_writeIrq);
        _writeIrqThread.irqFlag(_writeIrqFlag);
        //_writeIrqThread.clearIrq(_clearWriteIrq);
        // Master read interrupt thread connections
        _readIrqThread.irq(_readIrq);
        _readIrqThread.irqFlag(_readIrqFlag);
        //_readIrqThread.clearIrq(_clearReadIrq);
    }

    // Make sure these are connected to the slave interrupts in opposite order,
    // i.e. master write_interrupt connected to slave read_interrupt and
    // vice-versa.
    sc_in<bool> write_interrupt, read_interrupt;
    
    sc_out<bool> ready;
    sc_in <bool> ack;
    
    sc_out< sc_bv<ADDR_WIDTH> > A;
    sc_inout< sc_bv<DATA_WIDTH> > D;

    sc_port<os_api> os;

    void read ( void* data, unsigned long len )
    {
        this->_masterReadDriver.read(data, len);
    }

    void write ( const void* data, unsigned long len )
    {
        this->_masterWriteDriver.write(data, len);
    }

    private:

    MasterHardwareBus _master;
    MasterHardwareSyncDetect _writeIrq, _readIrq;
    MasterHardwareBusLinkAccess _masterMAC;
    kpn_MasterDriver _masterWriteDriver, _masterReadDriver;
    kpn_MasterInterruptThread _writeIrqThread, _readIrqThread;

    // Connections between the interrupt threads and master drivers
    sc_signal<bool> _writeIrqFlag, _readIrqFlag;
    sc_event _clearWriteIrq, _clearReadIrq;

};

#endif
