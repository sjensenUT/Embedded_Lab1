#include <assert.h>
#include "parser.h"

void	load(int lIdx, const char* attr, float* ptr, int size)
{
	char	fn[100];
	FILE*	fh;

	sprintf(fn, "out/l%i/%s.bin", lIdx, attr);
	fh = fopen(fn, "w");
	fread(ptr, sizeof(float), size, fh);
	fclose(fh);
}

int	main(int argc, char* argv[])
{
	char	 fn[100];
	FILE*	 fh;
	network* netPtr;

	// setup
	netPtr = parse_network_cfg("yolov2-tiny.cfg");

	// loop over layers
	for(int lIdx=0; lIdx<netPtr->n; lIdx++)
	{
		layer l = netPtr->layers[lIdx];

		if(l.type == CONVOLUTIONAL)
		{
			if(l.numload) l.n = l.numload;
			int num = l.c/l.groups*l.n*l.size*l.size;

			load(lIdx, "biases", l.biases, l.n);

			if(l.batch_normalize && (!l.dontloadscales))
			{
				load(lIdx, "scales", l.scales, l.n);
				load(lIdx, "mean",   l.rolling_mean, l.n);
				load(lIdx, "variance", l.rolling_variance, l.n);
			}

			load(lIdx, "weights", l.weights, num);

			printf("loaded parameters of layer %i\n", lIdx);
		}
	}
}
