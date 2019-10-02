#include <iostream>
#include <stdio.h>
using namespace std;

int* calcPrevCoords(int coords[], int stride, int filterSize, int prevWidth, int prevHeight, std::string prevLayerType){
        int x1 = coords[0];
	int y1 = coords[1];
	int x2 = coords[2];
	int y2 = coords[3];
	int *result = new int[4];
	if(prevLayerType.compare("convolutional") == 0){
		result[0] = max(0, stride*x1 - filterSize/2);
		result[1] = max(0, stride*y1 - filterSize/2);
		result[2] = min(stride*x2 + filterSize/2, prevWidth - 1);
		result[3] = min(stride*y2 + filterSize/2, prevHeight - 1);
	}else{
		result[0] = stride*x1;
                result[1] = stride*y1;
                result[2] = min(stride*x2 + stride - 1, prevWidth - 1);
                result[3] = min(stride*y2 + stride - 1, prevHeight - 1);
		//cout << "hello2" << endl;
	}	
	//cout << result[0] << ", " << result[1] << ", " << result[2] << ", " << result[3] << endl;
	return result;
	
}

float* getSubArray(float arr[], const int coords[], int width, int height, int numChannels){
    int x1 = coords[0];
    int y1 = coords[1];
    int x2 = coords[2];
    int y2 = coords[3];

//    cout << "getSubArray:" << endl;
//    printf("Coords (%d, %d) to (%d, %d), w = %d, h = %d\n",
//        x1, y1, x2, y2, width, height);
    float *result = new float[(x2 - x1 + 1)*(y2 - y1 + 1)*numChannels];
    int n = 0;
    for(int i = 0; i < numChannels; i++){
        for(int j = y1; j <= y2; j++){
            for(int k = x1; k <= x2; k++){
                result[n] = arr[i*width*height + j*width + k];
                n++;
            }
        }
    }
    return result;
}
                
/*
float* horzCat(float arr1[], float arr2[], int width1, int width2, int height, int numChannels){
	int totalWidth = width1 + width2;
	float *result = new float[totalWidth*height*numChannels];
	int n = 0;
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width1; j++){
			for(int k = 0; k < numChannels; k++){
				result[n] = arr1[(i*width1 + j)*numChannels + k];
				n++;
			} 	
		}
		for(int j = 0; j < width2; j++){
			for(int k = 0; k < numChannels; k++){
				result[n] = arr2[(i*width2 + j)*numChannels + k];
				n++;
			}
		}
	}
	return result;
}

float* vertCat(float arr1[], float arr2[], int width, int height1, int height2, int numChannels){
        int totalHeight = height1 + height2;
        float *result = new float[width*totalHeight*numChannels];
        int n = 0;
        for(int i = 0; i < height1; i++){
                for(int j = 0; j < width; j++){
                        for(int k = 0; k < numChannels; k++){
                                result[n] = arr1[(i*width + j)*numChannels + k];
                                n++;
                        }
                }
        }
	for(int i = 0; i < height2; i++){
                for(int j = 0; j < width; j++){
                        for(int k = 0; k < numChannels; k++){
                                result[n] = arr2[(i*width + j)*numChannels + k];
                                n++;
                        }
                }
        }
        return result;
}
*/
float* mergeTiles(float **tiles, const int widths[], const int heights[], int numChannels){
    //cout << "in mergeTiles" << endl;
    //cout << "tiles[0][0] = " << tiles[0][0] << endl;
    //cout << "calculating total width and height" << endl;
    int totalWidth = widths[0] + widths[1] + widths[2];
    int totalHeight = heights[0] + heights[1] + heights[2];
//    cout << "totalWidth = " << totalWidth << endl;
//    cout << "totalHeight = " << totalHeight << endl;
    float *result = new float[totalWidth*totalHeight*numChannels];
    int p = 0;
    //cout << "beginning tile merge" << endl;
    for(int i = 0; i < numChannels; i++){ // CHANNELS
	    for(int j = 0; j < 3; j++){ // TILE ROW
       	for(int k = 0; k < heights[j]; k++){ // Y COORDINATE
          for(int m = 0; m < 3; m++){ // TILE COL
				    for(int n = 0; n < widths[m]; n++){ // X COORDINATE
					    result[p] = tiles[3*j + m][i*heights[j]*widths[m] + k*widths[m] + n];
					    p++;
            }
          }
        }
      }
    }

    //cout << "tile merge complete" << endl;
    return result;
}

/*
int main() 
{
	int test[] = {0,0,4,4};
	//int *testResult = calcPrevCoords(test, 1, 3, 30, 30, "maxpool");
	//cout << testResult[0] << ", " << testResult[1] << ", " << testResult[2] << ", " << testResult[3] << endl;
	float *testFloats = new float[2700];
	int n = -1;
	for(int i = 0; i < 2700; i++){
		if(i%3 == 0){
			n++;
		}
		testFloats[i] = n;
	}
	
	float *testSubArray = getSubArray(testFloats, test, 30, 30, 3);
	for(int i = 0; i < 75; i++){
			if(i%15 == 0){
				cout << endl;
			}
			cout << testSubArray[i] << " ";	
	}
	cout << endl;
	
	float *testHorzCat = horzCat(testSubArray, testSubArray, 5, 5, 5, 3);
	for(int i = 0; i < 150; i++){
		if(i%30 ==  0){
			cout << endl;
		}
		cout << testHorzCat[i] << " ";	
	}
	cout << endl;
	float *testVertCat= vertCat(testSubArray, testSubArray, 5, 5, 5, 3);
	for(int i = 0; i < 150; i++){
		if(i%15 == 0){
			cout << endl;
		}
		cout << testVertCat[i] << " ";
	}
	cout << endl;


	
	float ** tiles = new float*[9];
	tiles[0] = new float[2] {1, 1};
	tiles[1] = new float[4] {2, 2, 3, 3};
	tiles[2] = new float[6] {4, 4, 5, 5, 6, 6};
	tiles[3] = new float[4] {7, 7, 8, 8};
	tiles[4] = new float[8] {9, 9, 10, 10, 11, 11, 12, 12};
	tiles[5] = new float[12] {13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18};
	tiles[6] = new float[6] {19, 19, 20, 20, 21, 21};
	tiles[7] = new float[12] {22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27};
	tiles[8] = new float[18] {28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36}; 
	int *widths = new int[3] {1, 2, 3};
	int *heights = new int[3] {1, 2, 3};

	float *mergeTest = mergeTiles(tiles, widths, heights, 2);
	for(int i = 0; i < 72; i++){
		if(i%12 == 0){
			cout << endl;
		}
		cout << mergeTest[i] << " ";
}

	return 0;
}*/
