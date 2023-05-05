// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-array-correlation-coefficient

#include <stdio.h>
#include <math.h>
#include <intrin.h>

// 汎用命令を使った、配列 a と b の相関係数を求める関数。
double correlation_coefficient_general(const int a[], const int b[], int length)
{
	int multiply_add = 0;

	int sum_a = 0;
	int sum_b = 0;

	int squared_sum_a = 0;
	int squared_sum_b = 0;

	for (int i = 0; i < length; i++)
	{
		multiply_add += a[i] * b[i];

		sum_a += a[i];
		sum_b += b[i];

		squared_sum_a += a[i] * a[i];
		squared_sum_b += b[i] * b[i];
	}

	// 平均を計算。
	double average_multiply = (double)multiply_add / length;

	double average_a = (double)sum_a / length;
	double average_b = (double)sum_b / length;

	double average_square_a = (double)squared_sum_a / length;
	double average_square_b = (double)squared_sum_b / length;

	// 分散を計算。
	double variance_a = average_square_a - (average_a * average_a);
	double variance_b = average_square_b - (average_b * average_b);

	// 共分散を計算。
	double covariance = average_multiply - (average_a * average_b);

	// 標準偏差を計算。
	double standard_deviation_a = sqrt(variance_a);
	double standard_deviation_b = sqrt(variance_b);

	return covariance / (standard_deviation_a * standard_deviation_b);
}

// SIMD 命令を使った、配列 a と b の相関係数を求める関数。
double correlation_coefficient(const int a[], const int b[], int length)
{
	int i = 0;

	// 合計値を 0 で初期化。
	__m256i multiply_add256 = _mm256_setzero_si256();

	__m256i sum_a256 = _mm256_setzero_si256();
	__m256i sum_b256 = _mm256_setzero_si256();

	__m256i squared_sum_a256 = _mm256_setzero_si256();
	__m256i squared_sum_b256 = _mm256_setzero_si256();

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i b256 = _mm256_loadu_si256((__m256i*)(&b[i]));

		__m256i multiply256 = _mm256_mullo_epi32(a256, b256);
		multiply_add256 = _mm256_add_epi32(multiply_add256, multiply256);

		sum_a256 = _mm256_add_epi32(sum_a256, a256);
		sum_b256 = _mm256_add_epi32(sum_b256, b256);

		__m256i squared_a256 = _mm256_mullo_epi32(a256, a256);
		__m256i squared_b256 = _mm256_mullo_epi32(b256, b256);

		squared_sum_a256 = _mm256_add_epi32(squared_sum_a256, squared_a256);
		squared_sum_b256 = _mm256_add_epi32(squared_sum_b256, squared_b256);
	}

	// 合計値をスカラー値に変換。
	// 汎用命令でも実効速度は変わらない。
	// https://stackoverflow.com/questions/42000693/why-my-avx2-horizontal-addition-function-is-not-faster-than-non-simd-addition
	__m256i multiply_add256_permute = _mm256_permute2x128_si256(multiply_add256, multiply_add256, 1);
	__m256i multiply_add256_hadd = _mm256_hadd_epi32(multiply_add256, multiply_add256_permute);
	multiply_add256_hadd = _mm256_hadd_epi32(multiply_add256_hadd, multiply_add256_hadd);
	multiply_add256_hadd = _mm256_hadd_epi32(multiply_add256_hadd, multiply_add256_hadd);
	int multiply_add = _mm256_extract_epi32(multiply_add256_hadd, 0);

	__m256i sum_a256_permute = _mm256_permute2x128_si256(sum_a256, sum_a256, 1);
	__m256i sum_a256_hadd = _mm256_hadd_epi32(sum_a256, sum_a256_permute);
	sum_a256_hadd = _mm256_hadd_epi32(sum_a256_hadd, sum_a256_hadd);
	sum_a256_hadd = _mm256_hadd_epi32(sum_a256_hadd, sum_a256_hadd);
	int sum_a = _mm256_extract_epi32(sum_a256_hadd, 0);

	__m256i sum_b256_permute = _mm256_permute2x128_si256(sum_b256, sum_b256, 1);
	__m256i sum_b256_hadd = _mm256_hadd_epi32(sum_b256, sum_b256_permute);
	sum_b256_hadd = _mm256_hadd_epi32(sum_b256_hadd, sum_b256_hadd);
	sum_b256_hadd = _mm256_hadd_epi32(sum_b256_hadd, sum_b256_hadd);
	int sum_b = _mm256_extract_epi32(sum_b256_hadd, 0);

	__m256i squared_sum_a256_permute = _mm256_permute2x128_si256(squared_sum_a256, squared_sum_a256, 1);
	__m256i squared_sum_a256_hadd = _mm256_hadd_epi32(squared_sum_a256, squared_sum_a256_permute);
	squared_sum_a256_hadd = _mm256_hadd_epi32(squared_sum_a256_hadd, squared_sum_a256_hadd);
	squared_sum_a256_hadd = _mm256_hadd_epi32(squared_sum_a256_hadd, squared_sum_a256_hadd);
	int squared_sum_a = _mm256_extract_epi32(squared_sum_a256_hadd, 0);

	__m256i squared_sum_b256_permute = _mm256_permute2x128_si256(squared_sum_b256, squared_sum_b256, 1);
	__m256i squared_sum_b256_hadd = _mm256_hadd_epi32(squared_sum_b256, squared_sum_b256_permute);
	squared_sum_b256_hadd = _mm256_hadd_epi32(squared_sum_b256_hadd, squared_sum_b256_hadd);
	squared_sum_b256_hadd = _mm256_hadd_epi32(squared_sum_b256_hadd, squared_sum_b256_hadd);
	int squared_sum_b = _mm256_extract_epi32(squared_sum_b256_hadd, 0);

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < length; ++i)
	{
		multiply_add += a[i] * b[i];

		sum_a += a[i];
		sum_b += b[i];

		squared_sum_a += a[i] * a[i];
		squared_sum_b += b[i] * b[i];
	}

	// 平均を計算。
	double average_multiply = (double)multiply_add / length;

	double average_a = (double)sum_a / length;
	double average_b = (double)sum_b / length;

	double average_square_a = (double)squared_sum_a / length;
	double average_square_b = (double)squared_sum_b / length;

	// 分散を計算。
	double variance_a = average_square_a - (average_a * average_a);
	double variance_b = average_square_b - (average_b * average_b);

	// 共分散を計算。
	double covariance = average_multiply - (average_a * average_b);

	// 標準偏差を計算。
	double standard_deviation_a = sqrt(variance_a);
	double standard_deviation_b = sqrt(variance_b);

	return covariance / (standard_deviation_a * standard_deviation_b);
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int b[] = { 3, 2, 1, 5, 9, 4, 6, 8, 7 };
	int length = sizeof(a) / sizeof(a[0]);

	double result = correlation_coefficient_general(a, b, length);
	printf("correlation_coefficient_general: %lf\n", result);

	result = correlation_coefficient(a, b, length);
	printf("correlation_coefficient:         %lf\n", result);

	return 0;
}
