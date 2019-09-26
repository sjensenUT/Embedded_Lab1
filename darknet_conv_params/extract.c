#include <assert.h>
#include "parser.h"

void	dump(int lIdx, const char* attr, float* ptr, int size)
{
	char	fn[100];
	FILE*	fh;

	sprintf(fn, "out/l%i/%s.bin", lIdx, attr);
	fh = fopen(fn, "w");
	fwrite(ptr, sizeof(float), size, fh);
	fclose(fh);
}

int	main(int argc, char* argv[])
{
	char	 fn[100];
	FILE*	 fh;
	network* netPtr;

	// setup
	netPtr = parse_network_cfg("yolov2-tiny.cfg");
	load_weights(netPtr, "yolov2-tiny.weights");

	// loop over layers
	for(int lIdx=0; lIdx<netPtr->n; lIdx++)
	{
		layer l = netPtr->layers[lIdx];

		if(l.type == CONVOLUTIONAL)
		{
			int num = l.c/l.groups*l.n*l.size*l.size;

			dump(lIdx, "biases", l.biases, l.n);

			if(l.batch_normalize && (!l.dontloadscales))
			{
				dump(lIdx, "scales", l.scales, l.n);
				dump(lIdx, "mean",   l.rolling_mean, l.n);
				dump(lIdx, "variance", l.rolling_variance, l.n);
			}

			dump(lIdx, "weights", l.weights, num);
		}
	}
}
