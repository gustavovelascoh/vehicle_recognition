/*
 * Bgsub.cpp
 *
 *  Created on: Jun 12, 2014
 *      Author: gustavo
 */

#include "Bgsub.h"

Bg_sub::Bg_sub() {
	// TODO Auto-generated constructor stub
	//fTau = 0.5;
}

Bg_sub::~Bg_sub() {
	// TODO Auto-generated destructor stub
}

float
Bg_sub::get_fTau() {
	std::cout << "xxx " << fTau << std::endl;
	return fTau;
}

int
Bg_sub::get_history()
{
	return history;
}

