#ifndef	KAHN_PROCESS_H
#define	KAHN_PROCESS_H
/*
 *	kahn_process.h -- Base SystemC module for modeling applications using KPN
 *
 *	System-Level Architecture and Modeling Lab
 *	Department of Electrical and Computer Engineering
 *	The University of Texas at Austin 
 *
 * 	Author: Kamyar Mirzazad Barijough (kammirzazad@utexas.edu)
 */

#include <systemc.h>
#include <chrono> 
using std::chrono::system_clock;
using std::chrono::milliseconds;

class	kahn_process : public sc_module
{
	public:

	SC_HAS_PROCESS(kahn_process);

	kahn_process(sc_module_name name) : sc_module(name)
	{
		iter = 0;
    max_iter = 0;
		SC_THREAD(main);
	}

	kahn_process(sc_module_name name, int _maxIter) : sc_module(name)
	{
		iter = 0;
    max_iter = _maxIter;
		SC_THREAD(main);
	}

	void	main()
    {   
       
      system_clock::time_point before = system_clock::now(); 
      
      init(); 
      while (iter < max_iter || max_iter == 0) {
          process(); 
          iter++;
      } 
      terminate();
      
      system_clock::time_point after = system_clock::now(); 
      milliseconds duration = std::chrono::duration_cast<milliseconds> (after - before); 

      cout << "SIMTIME: " << duration.count() << " ms." << endl;
    }
    //void  main()  { process(); }
	
	protected:

	int iter = 0;
    int max_iter;

	virtual void process() = 0;
    virtual void init() = 0;
    virtual void terminate() { }

};
#endif
