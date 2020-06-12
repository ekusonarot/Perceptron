#ifndef WEIGHT
#define WEIGHT

#include <fstream>
#include <stdlib.h>
#include <time.h>

void getweight(float* weight, int size, char* s)
{
	srand(time(NULL));
	std::fstream file(s, std::ios::binary | std::ios::in);

	if (!file) {
		for (int i = 0; i < size; i++) {
			weight[i] = (float)rand()/RAND_MAX;
		}
		file.close();
		return;
	}

	for (int i = 0; i < size; i++) {
		file >> weight[i];
	}
	file.close();
}

void saveweight(float* weight, int size, char* s)
{
	std::fstream file(s, std::ios::binary | std::ios::out);

	for (int i = 0; i < size; i++) {
		file << weight[i] << ' ';
	}
	file.close();
}

#endif // WEIGHT
