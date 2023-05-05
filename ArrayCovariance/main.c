// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-array-covariance

#include <stdio.h>
#include <intrin.h>

// �ėp���߂��g�����A�z�� a �� b �̋����U�����߂�֐��B
double covariance_general(const int a[], const int b[], int length)
{
	int multiply_add = 0;
	int sum_a = 0;
	int sum_b = 0;

	for (int i = 0; i < length; i++)
	{
		multiply_add += a[i] * b[i];
		sum_a += a[i];
		sum_b += b[i];
	}

	double average_multiply = (double)multiply_add / length;
	double average_a = (double)sum_a / length;
	double average_b = (double)sum_b / length;

	return average_multiply - (average_a * average_b);
}

// SIMD ���߂��g�����A�z�� a �� b �̋����U�����߂�֐��B
double covariance(const int a[], const int b[], int length)
{
	int i = 0;

	// ���v�l�� 0 �ŏ������B
	__m256i multiply_add256 = _mm256_setzero_si256();
	__m256i sum_a256 = _mm256_setzero_si256();
	__m256i sum_b256 = _mm256_setzero_si256();

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i b256 = _mm256_loadu_si256((__m256i*)(&b[i]));

		__m256i multiply256 = _mm256_mullo_epi32(a256, b256);
		multiply_add256 = _mm256_add_epi32(multiply_add256, multiply256);

		sum_a256 = _mm256_add_epi32(sum_a256, a256);
		sum_b256 = _mm256_add_epi32(sum_b256, b256);
	}

	// �X�J���[�l�ɕϊ��B
	// �ėp���߂ł��������x�͕ς��Ȃ��B
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

	// �c��̗v�f�������B
	for (; i < length; i++)
	{
		multiply_add += a[i] * b[i];
		sum_a += a[i];
		sum_b += b[i];
	}

	double average_multiply = (double)multiply_add / length;
	double average_a = (double)sum_a / length;
	double average_b = (double)sum_b / length;

	return average_multiply - (average_a * average_b);
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int b[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
	int length = sizeof(a) / sizeof(a[0]);

	double result = covariance_general(a, b, length);
	printf("covariance_general: %lf\n", result);

	result = covariance(a, b, length);
	printf("covariance:         %lf\n", result);

	return 0;
}
