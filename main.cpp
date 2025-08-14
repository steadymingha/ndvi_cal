// ndvi_fast.cpp
// Build: g++ ndvi_fast.cpp -O3 -march=native -fopenmp -o ndvi_fast `pkg-config --cflags --libs opencv4 libraw`
// Run (calibrate once):   ./ndvi_fast --calib /path/TS_RGB.dng /path/TS_NOIR.dng
// Run (fast runtime):     ./ndvi_fast --run   /path/TS_RGB.dng /path/TS_NOIR.dng
// Note: Comments in English only.

#include <opencv2/opencv.hpp>
#include <libraw/libraw.h>
#include <iostream>
#include <regex>
#include <filesystem>

using namespace std;
using namespace cv;

enum class Channel { RED, NIR };

static const Size CHESSBOARD_SIZE = Size(7, 10); // inner corners
static const double EPS = 1e-5;
static const string H_PATH = "homography.yml"; // cached H

// Extract timestamp from "YYYYMMDDHHMMSS_RGB.dng"
static string extractTS(const string& path) {
    smatch m; regex re(R"((\d{14})_(RGB|NOIR)\.dng$)");
    if (regex_search(path, m, re)) return m[1];
    return "OUT";
}

// Load small BGR for chessboard (fast demosaic)
static Mat loadBGRforCorners(const string& path, double scale=0.25) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) return Mat();
    if (raw.unpack() != LIBRAW_SUCCESS) return Mat();
    raw.imgdata.params.no_auto_bright = 1;
    raw.imgdata.params.use_camera_wb  = 1;
    if (raw.dcraw_process() != LIBRAW_SUCCESS) return Mat();
    libraw_processed_image_t* pim = raw.dcraw_make_mem_image();
    if (!pim || pim->colors != 3 || pim->bits != 8) {
        if (pim) LibRaw::dcraw_clear_mem(pim);
        raw.recycle(); return Mat();
    }
    Mat rgb(pim->height, pim->width, CV_8UC3, pim->data);
    Mat bgr; cvtColor(rgb, bgr, COLOR_RGB2BGR);
    Mat bgrClone = bgr.clone();
    LibRaw::dcraw_clear_mem(pim);
    raw.recycle();
    if (scale != 1.0) {
        Mat small; resize(bgrClone, small, Size(), scale, scale, INTER_AREA);
        return small;
    }
    return bgrClone;
}

// Get global/per-channel black level (LibRaw-robust)
static int getBlackLevel(LibRaw& raw) {
    int black = raw.imgdata.color.black;
    if (black <= 0) {
        int sum = 0, cnt = 0;
        for (int i = 0; i < 8; ++i) {
            int v = raw.imgdata.color.cblack[i];
            if (v > 0) { sum += v; cnt++; }
        }
        if (cnt > 0) black = sum / cnt;
    }
    return max(0, black);
}

// Load RAW half-res plane from Bayer (assume RGGB; R:(0,0), NIR uses (1,1))
static Mat loadRawChannel(const string& path, Channel ch) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) throw runtime_error("open failed: " + path);
    if (raw.unpack() != LIBRAW_SUCCESS) throw runtime_error("unpack failed: " + path);

    ushort* bayer = raw.imgdata.rawdata.raw_image;
    int h = raw.imgdata.sizes.raw_height;
    int w = raw.imgdata.sizes.raw_width;
    int pitch16 = raw.imgdata.sizes.raw_pitch / 2;
    int black = getBlackLevel(raw);

    Mat out(h/2, w/2, CV_32FC1);
    float* dst = out.ptr<float>(0);

    // OpenMP parallel loop for speed
    #pragma omp parallel for
    for (int y = 0; y < h; y += 2) {
        int oy = y/2;
        for (int x = 0; x < w; x += 2) {
            int ix = (ch == Channel::RED) ? x   : x+1;
            int iy = (ch == Channel::RED) ? y   : y+1;
            int idx = iy * pitch16 + ix;
            float val = float(max(0, int(bayer[idx]) - black));
            dst[oy * (w/2) + (x/2)] = val;
        }
    }
    raw.recycle();
    return out;
}

// Estimate H at downscale between NOIR and RGB (NOIR -> RGB)
static bool estimateH(const string& rgbPath, const string& noirPath, Mat& H_full, double scale=0.25) {
    Mat rgb_s = loadBGRforCorners(rgbPath, scale);
    Mat nir_s = loadBGRforCorners(noirPath, scale);
    if (rgb_s.empty() || nir_s.empty()) return false;

    Mat gL, gR; cvtColor(rgb_s, gL, COLOR_BGR2GRAY); cvtColor(nir_s, gR, COLOR_BGR2GRAY);
    vector<Point2f> cL, cR;
    bool okL = findChessboardCorners(gL, CHESSBOARD_SIZE, cL);
    bool okR = findChessboardCorners(gR, CHESSBOARD_SIZE, cR);
    if (!(okL && okR)) return false;

    cornerSubPix(gL, cL, Size(11,11), Size(-1,-1),
                 TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.001));
    cornerSubPix(gR, cR, Size(11,11), Size(-1,-1),
                 TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.001));

    // H at "scale" resolution
    Mat Hs = findHomography(cR, cL, RANSAC, 3.0);
    if (Hs.empty()) return false;

    // Lift H to full demosaiced resolution
    double sx = 1.0/scale, sy = 1.0/scale;
    Mat S  = (Mat_<double>(3,3) << sx,0,0, 0,sy,0, 0,0,1);
    Mat S_inv = (Mat_<double>(3,3) << 1.0/sx,0,0, 0,1.0/sy,0, 0,0,1);
    H_full = S * Hs * S_inv;
    return true;
}

// Load/save cached H (full demosaiced resolution)
static bool saveH(const Mat& H) {
    FileStorage fs(H_PATH, FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "H" << H;
    return true;
}
static bool loadH(Mat& H) {
    FileStorage fs(H_PATH, FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["H"] >> H;
    return !H.empty();
}

// Compute NDVI
static inline Mat computeNDVI(const Mat& nir, const Mat& red) {
    Mat denom, numer, ndvi;
    add(nir, red, denom);
    subtract(nir, red, numer);
    denom += EPS;
    divide(numer, denom, ndvi);
    ndvi.setTo(1.0f, ndvi > 1.0f);
    ndvi.setTo(-1.0f, ndvi < -1.0f);
    return ndvi;
}

// Save NDVI with fast PNG compression (or JPEG if preferred)
static void saveNDVIColored(const Mat& ndvi, const string& ts) {
    Mat ndvi_u8, colored;
    ndvi.convertTo(ndvi_u8, CV_8UC1, 127.5, 127.5);
    applyColorMap(ndvi_u8, colored, COLORMAP_JET);

    vector<int> params = {IMWRITE_PNG_COMPRESSION, 1}; // faster than default(3)
    imwrite("NDVI_" + ts + ".png", colored, params);
}

int main(int argc, char** argv) {
    // if (argc < 4) {
    //     cerr << "Usage:\n"
    //          << "  " << argv[0] << " --calib <RGB.dng> <NOIR.dng>\n"
    //          << "  " << argv[0] << " --run   <RGB.dng> <NOIR.dng>\n";
    //     return 1;
    // }
    // static const string RGB_DNG = "/home/user/ws/ndvi_cal/20250812_181606_RGB.dng";   // set your path
// static const string NOIR_DNG = "/home/user/ws/ndvi_cal/20250812_181606_NOIR.dng"; // set your path
    string rgbPath = "/home/user/ws/ndvi_cal/20250812_190639_RGB.dng";
    string noirPath = "/home/user/ws/ndvi_cal/20250812_190639_NOIR.dng";

    // string mode = string("--calib");

    // if (mode == string("--calib")) {
    //     Mat H_full;
    //     if (!estimateH(rgbPath, noirPath, H_full, 0.25)) {
    //         cerr << "Homography estimation failed.\n";
    //         return 2;
    //     }
    //     if (!saveH(H_full)) {
    //         cerr << "Save H failed.\n";
    //         return 3;
    //     }
    //     cout << "Saved H to " << H_PATH << endl;
    //     // return 0;
    // }
    string mode = string("--run");

    if (mode == string("--run")) {
        Mat H_full;
        if (!loadH(H_full)) {
            cerr << "No cached H. Run --calib first.\n";
            return 4;
        }

        // Scale H to RAW half-res grid
        // Need sizes to compute scale factors: read small BGR once (fast)
        Mat rgb_full_bgr = loadBGRforCorners(rgbPath, 1.0);
        if (rgb_full_bgr.empty()) { cerr << "Failed to read RGB for sizing.\n"; return 5; }

        Mat red_raw  = loadRawChannel(rgbPath,  Channel::RED);
        Mat nir_raw  = loadRawChannel(noirPath, Channel::NIR);

        double sx = double(red_raw.cols) / double(rgb_full_bgr.cols);
        double sy = double(red_raw.rows) / double(rgb_full_bgr.rows);
        Mat S  = (Mat_<double>(3,3) << sx,0,0, 0,sy,0, 0,0,1);
        Mat S_inv = (Mat_<double>(3,3) << 1.0/sx,0,0, 0,1.0/sy,0, 0,0,1);
        Mat H_half = S * H_full * S_inv;

        Mat nir_aligned;
        warpPerspective(nir_raw, nir_aligned, H_half, red_raw.size(), INTER_LINEAR, BORDER_CONSTANT);

        Mat ndvi = computeNDVI(nir_aligned, red_raw);
        string ts = extractTS(rgbPath);
        saveNDVIColored(ndvi, ts);
        cout << "Saved NDVI_" << ts << ".png\n";
        return 0;
    }

    cerr << "Unknown mode.\n";
    return 1;
}

