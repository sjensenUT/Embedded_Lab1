#include<string>
int* calcPrevCoords(int coords[], int stride, int filterSize, int prevWidth, int prevHeight, std::string prevLayerType);
//float* getSubArray(float arr[], int coords[], int width, int height, int numChannels);
//float* horzCat(float arr1[], float arr2[], int width1, int width2, int height, int numChannels);
//float* vertCat(float arr1[], float arr2[], int width, int height1, int height2, int numChannels);
float* mergeTiles(float **tiles, const int widths[], const int heights[], int numChannels);
