#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

void load_convolutional_weights(layer l, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
