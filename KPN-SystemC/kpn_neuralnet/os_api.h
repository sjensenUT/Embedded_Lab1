#ifndef	OS_API
#define	OS_API

/*
 *	os_api.h -- A minimal API for operating system model in SystemC
 *
 *	System-Level Architecture and Modeling Lab
 *	Department of Electrical and Computer Engineering
 *	The University of Texas at Austin 
 *
 * 	Author: Kamyar Mirzazad Barijough (kammirzazad@utexas.edu)
 */

#include <systemc.h>

class   os_api : virtual public sc_interface
{
	public:

	virtual int	pre_wait() = 0;
	virtual void	post_wait(int) = 0;
	virtual void	time_wait(int) = 0; // time in ms
	virtual void	task_terminate() = 0;
	virtual void	reg_task(const char*) = 0;
};

#endif