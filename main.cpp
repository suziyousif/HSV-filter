#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

#define TESTE
//#define WEBCAM
const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

int erosion_size = 0;
int elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;
Mat element_dilate, element_erode;
Mat labels;
Mat stats;
Mat centroids;

static void on_low_H_thresh_trackbar(int, void *);
static void on_high_H_thresh_trackbar(int, void *);
static void on_low_S_thresh_trackbar(int, void *);
static void on_high_S_thresh_trackbar(int, void *);
static void on_low_V_thresh_trackbar(int, void *);
static void on_high_V_thresh_trackbar(int, void *);
void Erosion( int, void* );
void Dilation( int, void* );


void createTrackbar_function(void);

#define HSV



int main(int argc, char* argv[])
{
    int i, n=0;

    #ifdef WEBCAM
      VideoCapture cap(argc > 1 ? atoi(argv[2]) : 0);

    #endif
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);

    // Trackbars to set thresholds 
    createTrackbar_function();
    
    // Default start
    Erosion( 0, 0 );
    Dilation( 0, 0 );

    Mat frame, frame_HSV, frame_threshold, frame_GRAY, frame_teste;
    Scalar color(255,0,0);

    while (true) {
      #ifdef WEBCAM
        cap >> frame;
        if(frame.empty())
        {
            break;
        }
      #else
        frame = imread("mms.jpg",1);
      #endif

      #ifdef HSV
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        #ifdef TESTE
          namedWindow("frame_HSV");
          imshow("frame_HSV", frame_threshold);
        #endif
      #endif


      //bitwise_not(frame_threshold,frame_threshold);
      #ifdef TESTE
      namedWindow("bitWiseNot");
      imshow("bitWiseNot", frame_threshold);
   
      #endif

      // apply the dilation and erosion operation
      dilate(frame_threshold, frame_threshold, element_dilate);
      #ifdef TESTE
      namedWindow("dilate");
      imshow("dilate", frame_threshold);
      #endif

      erode(frame_threshold, frame_threshold, element_erode);
      #ifdef TESTE
      namedWindow("erode");
      imshow("erode", frame_threshold);
      #endif

      // filter to remove noise
      medianBlur(frame_threshold,frame_threshold,15);
      #ifdef TESTE
      namedWindow("medianBlur");
      imshow("medianBlur", frame_threshold);
      #endif

      connectedComponentsWithStats(frame_threshold, labels, stats, centroids);
      bitwise_and(frame,frame ,frame_threshold );
      n=0;
      for(i=0; i<stats.rows; i++)
      {
        //cout<<stats.at<int>(Point(4, i))<<"\n";
        if(stats.at<int>(Point(4, i)) > 100  && stats.at<int>(Point(4, i)) < 20000){
          int x = stats.at<int>(Point(0, i));
          int y = stats.at<int>(Point(1, i));
          int w = stats.at<int>(Point(2, i));
          int h = stats.at<int>(Point(3, i));
          
          Rect rect(x,y,w,h);
          cv::rectangle(frame, rect, color);
          cv::rectangle(frame_threshold, rect, color);
          n++;
        }
      }
      
      putText(frame_threshold, std::to_string(n), Point(25,25), FONT_HERSHEY_DUPLEX, 1, color, 2);

      //show results
      imshow(window_capture_name, frame);
      imshow(window_detection_name, frame_threshold);
      char key = (char) waitKey(30);
      if (key == 'q' || key == 27)
      {
          imwrite( "image_edit_webcam.jpg", frame_threshold );
          break;
      }
    }
    return 0;
}


static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

void Erosion( int, void* )
{
  int erosion_type;
  if( elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( elem == 2) { erosion_type = MORPH_ELLIPSE; }

  element_erode = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
}

void Dilation( int, void* )
{
  int dilation_type;
  if( elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( elem == 2) { dilation_type = MORPH_ELLIPSE; }

  element_dilate = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size +1 ),
                                       Point( dilation_size , dilation_size ) );
}

void createTrackbar_function(void){
  
  createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
  createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
  createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
  createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
  createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
  createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", window_detection_name, &elem, max_elem, Erosion );

  createTrackbar( "dilation:Kernel size:\n 2n +1", window_detection_name, &dilation_size, max_kernel_size, Dilation );

  createTrackbar( "erosion:Kernel size:\n 2n +1", window_detection_name, &erosion_size, max_kernel_size, Erosion );
  
}
//links utilizados para o desenvolvimento
//https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
//https://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
//https://medium.com/analytics-vidhya/images-processing-segmentation-and-objects-counting-in-an-image-with-python-and-opencv-216cd38aca8e