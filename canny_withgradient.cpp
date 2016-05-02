
#include<opencv2/opencv.hpp>
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <vector>
#include <iostream>

using namespace cv;


void Cannywgradient(OutputArray _dst,
	double low_thresh, double high_thresh,
	bool L2gradient, InputOutputArray _dx, InputOutputArray _dy)
{
	const int type = _dx.type(), depth = CV_MAT_DEPTH(type), cn = 1;
	const Size size = _dx.size();

	CV_Assert(depth == CV_16S);
	_dst.create(size, CV_8U);

	if (!L2gradient && (CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
	{
		// backward compatibility
		L2gradient = true;
	}


	if (low_thresh > high_thresh)
		std::swap(low_thresh, high_thresh);

	Mat dst = _dst.getMat();
	Mat dx, dy;
	dx = _dx.getMat(), dy = _dy.getMat();
	if (L2gradient)
	{
		low_thresh = std::min(32767.0, low_thresh);
		high_thresh = std::min(32767.0, high_thresh);

		if (low_thresh > 0) low_thresh *= low_thresh;
		if (high_thresh > 0) high_thresh *= high_thresh;
	}
	int low = cvFloor(low_thresh);
	int high = cvFloor(high_thresh);

	ptrdiff_t mapstep = dx.cols + 2;
	AutoBuffer<uchar> buffer((dx.cols + 2)*(dx.rows + 2) + cn * mapstep * 3 * sizeof(int));

	int* mag_buf[3];
	mag_buf[0] = (int*)(uchar*)buffer;
	mag_buf[1] = mag_buf[0] + mapstep*cn;
	mag_buf[2] = mag_buf[1] + mapstep*cn;
	memset(mag_buf[0], 0, /* cn* */mapstep * sizeof(int));

	uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
	memset(map, 1, mapstep);
	memset(map + mapstep*(dx.rows + 1), 1, mapstep);

	int maxsize = std::max(1 << 10, dx.cols * dx.rows / 10);
	std::vector<uchar*> stack(maxsize);
	uchar **stack_top = &stack[0];
	uchar **stack_bottom = &stack[0];

	/* sector numbers
	(Top-Left Origin)

	1   2   3
	*  *  *
	* * *
	0*******0
	* * *
	*  *  *
	3   2   1
	*/

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

#if CV_SSE2
	bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

	// calculate magnitude and angle of gradient, perform non-maxima suppression.
	// fill the map with one of the following values:
	//   0 - the pixel might belong to an edge
	//   1 - the pixel can not belong to an edge
	//   2 - the pixel does belong to an edge
	for (int i = 0; i <= dx.rows; i++)
	{
		int* _norm = mag_buf[(i > 0) + 1] + 1;
		if (i < dx.rows)
		{
			short* _dx = dx.ptr<short>(i);
			short* _dy = dy.ptr<short>(i);

			if (!L2gradient)
			{
				int j = 0, width = dx.cols * cn;
#if CV_SSE2
				if (haveSSE2)
				{
					__m128i v_zero = _mm_setzero_si128();
					for (; j <= width - 8; j += 8)
					{
						__m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
						__m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));
						v_dx = _mm_max_epi16(v_dx, _mm_sub_epi16(v_zero, v_dx));
						v_dy = _mm_max_epi16(v_dy, _mm_sub_epi16(v_zero, v_dy));

						__m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx, v_zero), _mm_unpacklo_epi16(v_dy, v_zero));
						_mm_storeu_si128((__m128i *)(_norm + j), v_norm);

						v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx, v_zero), _mm_unpackhi_epi16(v_dy, v_zero));
						_mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
					}
				}
#elif CV_NEON
				for (; j <= width - 8; j += 8)
				{
					int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
					vst1q_s32(_norm + j, vaddq_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_dx))),
						vabsq_s32(vmovl_s16(vget_low_s16(v_dy)))));
					vst1q_s32(_norm + j + 4, vaddq_s32(vabsq_s32(vmovl_s16(vget_high_s16(v_dx))),
						vabsq_s32(vmovl_s16(vget_high_s16(v_dy)))));
				}
#endif
				for (; j < width; ++j)
					_norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
			}
			else
			{
				int j = 0, width = dx.cols * cn;
#if CV_SSE2
				if (haveSSE2)
				{
					for (; j <= width - 8; j += 8)
					{
						__m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
						__m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

						__m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
						__m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

						__m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
						_mm_storeu_si128((__m128i *)(_norm + j), v_norm);

						v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
						_mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
					}
				}
#elif CV_NEON
				for (; j <= width - 8; j += 8)
				{
					int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
					int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
					int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
					vst1q_s32(_norm + j, v_dst);

					v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
					v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
					vst1q_s32(_norm + j + 4, v_dst);
				}
#endif
				for (; j < width; ++j)
					_norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
			}

			if (cn > 1)
			{
				for (int j = 0, jn = 0; j < dx.cols; ++j, jn += cn)
				{
					int maxIdx = jn;
					for (int k = 1; k < cn; ++k)
						if (_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
					_norm[j] = _norm[maxIdx];
					_dx[j] = _dx[maxIdx];
					_dy[j] = _dy[maxIdx];
				}
			}
			_norm[-1] = _norm[dx.cols] = 0;
		}
		else
			memset(_norm - 1, 0, /* cn* */mapstep * sizeof(int));

		// at the very beginning we do not have a complete ring
		// buffer of 3 magnitude rows for non-maxima suppression
		if (i == 0)
			continue;

		uchar* _map = map + mapstep*i + 1;
		_map[-1] = _map[dx.cols] = 1;

		int* _mag = mag_buf[1] + 1; // take the central row
		ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
		ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

		const short* _x = dx.ptr<short>(i - 1);
		const short* _y = dy.ptr<short>(i - 1);

		if ((stack_top - stack_bottom) + dx.cols > maxsize)
		{
			int sz = (int)(stack_top - stack_bottom);
			maxsize = std::max(maxsize * 3 / 2, sz + dx.cols);
			stack.resize(maxsize);
			stack_bottom = &stack[0];
			stack_top = stack_bottom + sz;
		}

		int prev_flag = 0;
		for (int j = 0; j < dx.cols; j++)
		{
#define CANNY_SHIFT 15
			const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

			int m = _mag[j];

			if (m > low)
			{
				int xs = _x[j];
				int ys = _y[j];
				int x = std::abs(xs);
				int y = std::abs(ys) << CANNY_SHIFT;

				int tg22x = x * TG22;

				if (y < tg22x)
				{
					if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
				}
				else
				{
					int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
					if (y > tg67x)
					{
						if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
					}
					else
					{
						int s = (xs ^ ys) < 0 ? -1 : 1;
						if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
					}
				}
			}
			prev_flag = 0;
			_map[j] = uchar(1);
			continue;
		__ocv_canny_push:
			if (!prev_flag && m > high && _map[j - mapstep] != 2)
			{
				CANNY_PUSH(_map + j);
				prev_flag = 1;
			}
			else
				_map[j] = 0;
		}

		// scroll the ring buffer
		_mag = mag_buf[0];
		mag_buf[0] = mag_buf[1];
		mag_buf[1] = mag_buf[2];
		mag_buf[2] = _mag;
	}

	// now track the edges (hysteresis thresholding)
	while (stack_top > stack_bottom)
	{
		uchar* m;
		if ((stack_top - stack_bottom) + 8 > maxsize)
		{
			int sz = (int)(stack_top - stack_bottom);
			maxsize = maxsize * 3 / 2;
			stack.resize(maxsize);
			stack_bottom = &stack[0];
			stack_top = stack_bottom + sz;
		}

		CANNY_POP(m);

		if (!m[-1])         CANNY_PUSH(m - 1);
		if (!m[1])          CANNY_PUSH(m + 1);
		if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
		if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
		if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
		if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
		if (!m[mapstep])    CANNY_PUSH(m + mapstep);
		if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
	}
	// the final pass, form the final image
	const uchar* pmap = map + mapstep + 1;
	uchar* pdst = dst.ptr();
	for (int i = 0; i < dx.rows; i++, pmap += mapstep, pdst += dst.step)
	{
		for (int j = 0; j < dx.cols; j++)
			pdst[j] = (uchar)-(pmap[j] >> 1);
	}
}




