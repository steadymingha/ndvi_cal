// ndvi_pipeline.cpp
// Build:
//   g++ ndvi_pipeline.cpp -O3 -march=native -fopenmp -o ndvi_pipeline `pkg-config --cflags --libs opencv4 libraw`
// Usage:
//   ./ndvi_pipeline dark    <RGB_dark.dng> <NOIR_dark.dng>                   <out_dir>
//   ./ndvi_pipeline white   <RGB_white.dng> <NOIR_white.dng>                 <out_dir> <Rw>
//   ./ndvi_pipeline calib   <RGB_chess.dng> <NOIR_chess.dng>                 <out_dir>
//   ./ndvi_pipeline capture <RGB.dng>      <NOIR.dng>                        <dir_with_calib> [out.png]
//
// Notes:
// - CFA assumed BGGR (Blue,Green / Green,Red). Red site = (x+1, y+1).
// - Radiometry: dark/white based, exposure-normalized with ISO, shutter, aperture (f-number).
// - Homography saved as H_half (RAW half-res grid). No scaling in capture.
// - Comments: English only.

#include <opencv2/opencv.hpp>
#include <libraw/libraw.h>
#include <iostream>
#include <filesystem>
#include <regex>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// ---------- Config ----------
static const Size  CB_SIZE = Size(9,6);   // inner corners
static const double EPS = 1e-6;
static const double DOWNSCALE = 0.35;     // chessboard detection scale

enum class Channel { RED };

// ---------- Utilities ----------
static bool ensureDir(const string& dir) {
    std::error_code ec; fs::create_directories(dir, ec); return !ec;
}
static string tsFromName(const string& path) {
    smatch m; regex re(R"((\d{14})_(RGB|NOIR)\.dng$)");
    if (regex_search(path, m, re)) return m[1];
    return "OUT";
}

// ---------- Shot metadata (EXIF-like) ----------
struct ShotMeta {
    double iso = 0.0;         // ISO speed
    double shutter_s = 0.0;   // seconds
    double aperture = 0.0;    // f-number
};

static ShotMeta readShotMeta(const string& dngPath) {
    ShotMeta sm;
    LibRaw raw;
    if (raw.open_file(dngPath.c_str()) == LIBRAW_SUCCESS) {
        raw.unpack();
        sm.iso       = raw.imgdata.other.iso_speed;
        sm.shutter_s = raw.imgdata.other.shutter;   // seconds
        sm.aperture  = raw.imgdata.other.aperture;  // f-number
        raw.recycle();
    }
    return sm;
}

static inline double safeISO(const ShotMeta& m)     { return (m.iso       > 0) ? m.iso       : 100.0; }
static inline double safeT(const ShotMeta& m)       { return (m.shutter_s > 0) ? m.shutter_s : 0.01; }
static inline double safeF(const ShotMeta& m)       { return (m.aperture  > 0) ? m.aperture  : 2.8; }

// Exposure scale with aperture: E ∝ (t * ISO) / f^2
static double exposureScaleRobust(const ShotMeta& ref, const ShotMeta& cur) {
    const double E_ref = (safeT(ref) * safeISO(ref)) / (safeF(ref)*safeF(ref));
    const double E_cur = (safeT(cur) * safeISO(cur)) / (safeF(cur)*safeF(cur));
    return (E_ref > 0) ? (E_ref / std::max(EPS, E_cur)) : 1.0;
}

// ---------- LibRaw helpers ----------
static int getBlack(LibRaw& raw) {
    int black = raw.imgdata.color.black;
    if (black <= 0) {
        int sum=0, cnt=0;
        for (int i=0;i<8;++i) { int v = raw.imgdata.color.cblack[i]; if (v>0){ sum+=v; cnt++; } }
        if (cnt>0) black = sum/cnt;
    }
    return std::max(0, black);
}

static pair<int,int> rawHalfSize(const string& dng) {
    LibRaw r; r.open_file(dng.c_str()); r.unpack();
    int w = r.imgdata.sizes.raw_width;
    int h = r.imgdata.sizes.raw_height;
    r.recycle();
    return {w/2, h/2};
}

// Load RAW half-res plane from BGGR red sites (x+1,y+1)
static Mat loadRawRedLike(const string& path) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) throw runtime_error("open failed: "+path);
    if (raw.unpack() != LIBRAW_SUCCESS)               throw runtime_error("unpack failed: "+path);

    ushort* bayer = raw.imgdata.rawdata.raw_image;
    const int h = raw.imgdata.sizes.raw_height;
    const int w = raw.imgdata.sizes.raw_width;
    const int stride = raw.imgdata.sizes.raw_pitch/2;
    const int black = getBlack(raw);

    Mat out(h/2, w/2, CV_32FC1);
    float* dst = out.ptr<float>(0);

    #pragma omp parallel for schedule(static)
    for (int y=0; y<h; y+=2) {
        const int oy = y/2;
        float* row_o = dst + oy*(w/2);
        for (int x=0; x<w; x+=2) {
            const int ix = x+1, iy = y+1;         // red site for BGGR
            const int idx = iy*stride + ix;
            const int dn  = std::max(0, int(bayer[idx]) - black);
            row_o[x/2] = float(dn);
        }
    }
    raw.recycle();
    return out;
}

// Demosaic to BGR for chessboard corner detection (only in calib)
static Mat loadBGRforCorners(const string& path, double scale=DOWNSCALE) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) return Mat();
    if (raw.unpack() != LIBRAW_SUCCESS) return Mat();
    raw.imgdata.params.no_auto_bright = 1;
    raw.imgdata.params.use_camera_wb  = 1;
    if (raw.dcraw_process() != LIBRAW_SUCCESS) return Mat();
    libraw_processed_image_t* pim = raw.dcraw_make_mem_image();
    if (!pim || pim->colors!=3 || pim->bits!=8) { if (pim) LibRaw::dcraw_clear_mem(pim); raw.recycle(); return Mat(); }
    Mat rgb(pim->height, pim->width, CV_8UC3, pim->data);
    Mat bgr; cvtColor(rgb, bgr, COLOR_RGB2BGR);
    Mat out = bgr.clone();
    LibRaw::dcraw_clear_mem(pim);
    raw.recycle();
    if (scale!=1.0) { Mat small; resize(out, small, Size(), scale, scale, INTER_AREA); return small; }
    return out;
}

// ---------- Radiometric calibration ----------
struct RadCalib {
    Mat B_red,  B_nir;     // dark frames (half-res, float)
    Mat WmB_red, WmB_nir;  // (white - dark) (half-res, float)
    double Rw = 1.0;       // panel reflectance
    ShotMeta ref_red, ref_nir; // exposure of white frames
};

static bool saveRad(const string& dir, const RadCalib& C) {
    FileStorage fs(dir + "/radiometric.yml", FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "Rw" << C.Rw;
    fs << "B_red" << C.B_red;
    fs << "B_nir" << C.B_nir;
    fs << "WmB_red" << C.WmB_red;
    fs << "WmB_nir" << C.WmB_nir;
    fs << "ref_iso_red" << C.ref_red.iso << "ref_t_red" << C.ref_red.shutter_s << "ref_f_red" << C.ref_red.aperture;
    fs << "ref_iso_nir" << C.ref_nir.iso << "ref_t_nir" << C.ref_nir.shutter_s << "ref_f_nir" << C.ref_nir.aperture;
    return true;
}
static bool loadRad(const string& dir, RadCalib& C) {
    FileStorage fs(dir + "/radiometric.yml", FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["Rw"] >> C.Rw;
    fs["B_red"] >> C.B_red;
    fs["B_nir"] >> C.B_nir;
    fs["WmB_red"] >> C.WmB_red;
    fs["WmB_nir"] >> C.WmB_nir;
    fs["ref_iso_red"] >> C.ref_red.iso;   fs["ref_t_red"] >> C.ref_red.shutter_s; fs["ref_f_red"] >> C.ref_red.aperture;
    fs["ref_iso_nir"] >> C.ref_nir.iso;   fs["ref_t_nir"] >> C.ref_nir.shutter_s; fs["ref_f_nir"] >> C.ref_nir.aperture;
    return true;
}

// Safe divide (avoid global EPS add)
static void safeDivide(const Mat& num, const Mat& den, Mat& out) {
    out.create(num.size(), CV_32FC1);
    #pragma omp parallel for schedule(static)
    for (int y=0; y<num.rows; ++y) {
        const float* pn = num.ptr<float>(y);
        const float* pd = den.ptr<float>(y);
        float* po = out.ptr<float>(y);
        for (int x=0; x<num.cols; ++x) {
            const float d = pd[x];
            po[x] = (d > 0.f) ? (pn[x] / d) : 0.f;
        }
    }
}

// ---------- Homography (store as H_half) ----------
static bool estimateH_half(const string& rgbPath, const string& nirPath, Mat& H_half) {
    // 1) H at downscaled demosaiced resolution
    Mat rgb_s  = loadBGRforCorners(rgbPath, DOWNSCALE);
    Mat nir_s  = loadBGRforCorners(nirPath, DOWNSCALE);
    if (rgb_s.empty() || nir_s.empty()) return false;

    Mat gL, gR; cvtColor(rgb_s, gL, COLOR_BGR2GRAY); cvtColor(nir_s, gR, COLOR_BGR2GRAY);
    vector<Point2f> cL, cR;
    bool okL = findChessboardCorners(gL, CB_SIZE, cL);
    bool okR = findChessboardCorners(gR, CB_SIZE, cR);
    if (!(okL && okR)) return false;

    cornerSubPix(gL, cL, Size(11,11), Size(-1,-1),
        TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.001));
    cornerSubPix(gR, cR, Size(11,11), Size(-1,-1),
        TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER, 30, 0.001));

    Mat Hs = findHomography(cR, cL, RANSAC, 3.0);
    if (Hs.empty()) return false;

    // 2) Lift to full demosaiced size
    // Obtain full demosaic dims once (calibration time only)
    LibRaw tmp; tmp.open_file(rgbPath.c_str()); tmp.unpack();
    tmp.imgdata.params.no_auto_bright=1; tmp.imgdata.params.use_camera_wb=1; tmp.dcraw_process();
    libraw_processed_image_t* pim = tmp.dcraw_make_mem_image();
    int full_w = pim->width, full_h = pim->height;
    LibRaw::dcraw_clear_mem(pim); tmp.recycle();

    const double sx_full = 1.0/DOWNSCALE, sy_full = 1.0/DOWNSCALE;
    Mat S1 = (Mat_<double>(3,3) << sx_full,0,0, 0,sy_full,0, 0,0,1);
    Mat H_full = S1 * Hs * S1.inv();

    // 3) Convert to RAW half-res grid (use LibRaw raw size)
    LibRaw rraw; rraw.open_file(rgbPath.c_str()); rraw.unpack();
    int raw_w = rraw.imgdata.sizes.raw_width;
    int raw_h = rraw.imgdata.sizes.raw_height;
    rraw.recycle();

    const double sx_half = (raw_w/2.0) / double(full_w);
    const double sy_half = (raw_h/2.0) / double(full_h);
    Mat S2 = (Mat_<double>(3,3) << sx_half,0,0, 0,sy_half,0, 0,0,1);
    H_half = S2 * H_full * S2.inv(); // final H in RAW half-res coordinates
    return true;
}

static bool saveH(const string& dir, const Mat& H_half) {
    FileStorage fs(dir + "/H.yml", FileStorage::WRITE);
    if (!fs.isOpened()) return false; fs << "H_half" << H_half; return true;
}
static bool loadH(const string& dir, Mat& H_half) {
    FileStorage fs(dir + "/H.yml", FileStorage::READ);
    if (!fs.isOpened()) return false; fs["H_half"] >> H_half; return !H_half.empty();
}

// ---------- NDVI ----------
static inline void ndviParallel(const Mat& NIR, const Mat& RED, Mat& NDVI) {
    NDVI.create(NIR.size(), CV_32FC1);
    #pragma omp parallel for schedule(static)
    for (int y=0; y<NIR.rows; ++y) {
        const float* pn = NIR.ptr<float>(y);
        const float* pr = RED.ptr<float>(y);
        float* pd = NDVI.ptr<float>(y);
        for (int x=0; x<NIR.cols; ++x) {
            const float n = pn[x], r = pr[x];
            const float den = n + r;
            float v = (den > 0.f) ? (n - r) / den : 0.f;
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            pd[x] = v;
        }
    }
}

static void saveColored(const Mat& ndvi, const string& outPNG) {
    Mat u8, colored;
    ndvi.convertTo(u8, CV_8UC1, 127.5, 127.5);
    applyColorMap(u8, colored, COLORMAP_JET);
    vector<int> p = {IMWRITE_PNG_COMPRESSION, 1}; // faster
    imwrite(outPNG, colored, p);
}

// ---------- Modes ----------
static int mode_dark(const string& rgb_dark, const string& nir_dark, const string& outDir) {
    if (!ensureDir(outDir)) { cerr << "cannot create dir\n"; return 2; }
    fs::copy_file(rgb_dark, outDir+"/dark_red.dng", fs::copy_options::overwrite_existing);
    fs::copy_file(nir_dark, outDir+"/dark_nir.dng", fs::copy_options::overwrite_existing);
    cout << "dark frames stored in " << outDir << "\n";
    return 0;
}

static int mode_white(const string& rgb_white, const string& nir_white, const string& outDir, double Rw) {
    if (!ensureDir(outDir)) { cerr << "cannot create dir\n"; return 2; }

    const string dr = outDir + "/dark_red.dng";
    const string dn = outDir + "/dark_nir.dng";
    if (!fs::exists(dr) || !fs::exists(dn)) { cerr << "dark frames not found\n"; return 3; }

    RadCalib C; C.Rw = (Rw>0 && Rw<=1.0) ? Rw : 1.0;

    C.B_red = loadRawRedLike(dr);
    C.B_nir = loadRawRedLike(dn);
    Mat W_red = loadRawRedLike(rgb_white);
    Mat W_nir = loadRawRedLike(nir_white);
    if (C.B_red.size()!=W_red.size() || C.B_nir.size()!=W_nir.size()) { cerr << "size mismatch\n"; return 4; }

    C.ref_red = readShotMeta(rgb_white);
    C.ref_nir = readShotMeta(nir_white);

    subtract(W_red, C.B_red, C.WmB_red);
    subtract(W_nir, C.B_nir, C.WmB_nir);

    // Protect zeros only (optional threshold)
    // Here we just ensure non-negative.
    threshold(C.WmB_red, C.WmB_red, 0.f, 0.f, THRESH_TOZERO);
    threshold(C.WmB_nir, C.WmB_nir, 0.f, 0.f, THRESH_TOZERO);

    fs::copy_file(rgb_white, outDir+"/white_red.dng", fs::copy_options::overwrite_existing);
    fs::copy_file(nir_white, outDir+"/white_nir.dng", fs::copy_options::overwrite_existing);

    if (!saveRad(outDir, C)) { cerr << "save radiometric failed\n"; return 5; }
    cout << "radiometric.yml saved in " << outDir << "\n";
    return 0;
}

static int mode_calib(const string& rgb_chess, const string& nir_chess, const string& outDir) {
    if (!ensureDir(outDir)) { cerr << "cannot create dir\n"; return 2; }
    Mat Hh;
    if (!estimateH_half(rgb_chess, nir_chess, Hh)) { cerr << "homography estimation failed\n"; return 3; }
    if (!saveH(outDir, Hh)) { cerr << "save H failed\n"; return 4; }
    cout << "H.yml (H_half) saved in " << outDir << "\n";
    return 0;
}

static int mode_capture(const string& rgb_path, const string& nir_path, const string& dir, const string& outPNG_opt) {
    RadCalib C;
    if (!loadRad(dir, C)) { cerr << "load radiometric failed\n"; return 2; }
    Mat H_half;
    if (!loadH(dir, H_half)) { cerr << "load H failed\n"; return 3; }

    // Load RAW planes
    Mat R_raw = loadRawRedLike(rgb_path);
    Mat N_raw = loadRawRedLike(nir_path);
    if (R_raw.size() != C.B_red.size() || N_raw.size() != C.B_nir.size()) { cerr << "size mismatch to calib\n"; return 4; }

    // Dark subtract on sensor grids
    Mat RmB, NmB;
    subtract(R_raw, C.B_red, RmB);
    subtract(N_raw, C.B_nir, NmB);

    // Warp NIR-side (NmB and WmB_nir) to RED grid using H_half
    Mat Nmb_aligned, WmB_nir_aligned;
    warpPerspective(NmB,       Nmb_aligned,      H_half, RmB.size(), INTER_LINEAR, BORDER_REPLICATE);
    warpPerspective(C.WmB_nir, WmB_nir_aligned,  H_half, RmB.size(), INTER_LINEAR, BORDER_REPLICATE);

    // Exposure normalization (to white reference exposure), with aperture
    const ShotMeta cur_red = readShotMeta(rgb_path);
    const ShotMeta cur_nir = readShotMeta(nir_path);
    const double s_red = exposureScaleRobust(C.ref_red, cur_red);
    const double s_nir = exposureScaleRobust(C.ref_nir, cur_nir);

    // Reflectance on the same grid: (I-B)/(W-B) * Rw * s
    Mat R_reflect, N_reflect;
    safeDivide(RmB,        C.WmB_red,       R_reflect);
    safeDivide(Nmb_aligned,WmB_nir_aligned, N_reflect);
    R_reflect *= (C.Rw * s_red);
    N_reflect *= (C.Rw * s_nir);

    // Clamp negatives to zero
    threshold(R_reflect, R_reflect, 0.f, 0.f, THRESH_TOZERO);
    threshold(N_reflect, N_reflect, 0.f, 0.f, THRESH_TOZERO);

    // NDVI
    Mat NDVI;
    ndviParallel(N_reflect, R_reflect, NDVI);

    // Save
    const string outPNG = outPNG_opt.empty() ? ("NDVI_"+tsFromName(rgb_path)+".png") : outPNG_opt;
    Mat u8, colored;
    NDVI.convertTo(u8, CV_8UC1, 127.5, 127.5);
    applyColorMap(u8, colored, COLORMAP_JET);
    vector<int> p = {IMWRITE_PNG_COMPRESSION, 1};
    imwrite(outPNG, colored, p);
    cout << "Saved " << outPNG << "\n";
    return 0;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " dark    <RGB_dark.dng> <NOIR_dark.dng>                   <out_dir>\n"
             << "  " << argv[0] << " white   <RGB_white.dng> <NOIR_white.dng>                 <out_dir> <Rw>\n"
             << "  " << argv[0] << " calib   <RGB_chess.dng> <NOIR_chess.dng>                 <out_dir>\n"
             << "  " << argv[0] << " capture <RGB.dng>      <NOIR.dng>                        <dir_with_calib> [out.png]\n";
        return 1;
    }

    const string mode = argv[1];

    if (mode == "dark") {
        if (argc < 5) { cerr << "args\n"; return 1; }
        return mode_dark(argv[2], argv[3], argv[4]);
    } else if (mode == "white") {
        if (argc < 6) { cerr << "args\n"; return 1; }
        const double Rw = atof(argv[5]);
        return mode_white(argv[2], argv[3], argv[4], Rw);
    } else if (mode == "calib") {
        if (argc < 5) { cerr << "args\n"; return 1; }
        return mode_calib(argv[2], argv[3], argv[4]);
    } else if (mode == "capture") {
        if (argc < 5) { cerr << "args\n"; return 1; }
        const string outPNG = (argc >= 6) ? argv[5] : "";
        return mode_capture(argv[2], argv[3], argv[4], outPNG);
    } else {
        cerr << "unknown mode\n";
        return 1;
    }
}
