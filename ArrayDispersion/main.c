// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-array-dispersion

#include <stdio.h>
#include <intrin.h>

// �ėp���߂��g�����A�z�� a �̕��U�����߂�֐��B
double dispersion_general(const int a[], int length)
{
    int sum = 0;
    int squared_sum = 0;

    for (int i = 0; i < length; i++)
    {
        sum += a[i];
        squared_sum += a[i] * a[i];
    }

    double average = (double)sum / length;
    double squred_average = (double)squared_sum / length;

    return squred_average - (average * average);
}

// SIMD ���߂��g�����A�z�� a �̕��U�����߂�֐��B
double dispersion(const int a[], int length)
{
	int i = 0;

	// ���v�l�� 0 �ŏ������B
	__m256i sum256 = _mm256_setzero_si256();
	__m256i squared_sum256 = _mm256_setzero_si256();

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));

		sum256 = _mm256_add_epi32(sum256, a256);

		__m256i squared_a256 = _mm256_mullo_epi32(a256, a256);
		squared_sum256 = _mm256_add_epi32(squared_sum256, squared_a256);
	}

	// ���v�l���X�J���[�l�ɕϊ��B
	// �ėp���߂ł��������x�͕ς��Ȃ��B
	// https://stackoverflow.com/questions/42000693/why-my-avx2-horizontal-addition-function-is-not-faster-than-non-simd-addition
	__m256i sum256_permute = _mm256_permute2x128_si256(sum256, sum256, 1);
	__m256i result256 = _mm256_hadd_epi32(sum256, sum256_permute);
	result256 = _mm256_hadd_epi32(result256, result256);
	result256 = _mm256_hadd_epi32(result256, result256);
	int sum = _mm256_extract_epi32(result256, 0);

	__m256i squared_sum256_permute = _mm256_permute2x128_si256(squared_sum256, squared_sum256, 1);
	__m256i squared_result256 = _mm256_hadd_epi32(squared_sum256, squared_sum256_permute);
	squared_result256 = _mm256_hadd_epi32(squared_result256, squared_result256);
	squared_result256 = _mm256_hadd_epi32(squared_result256, squared_result256);
	int squared_sum = _mm256_extract_epi32(squared_result256, 0);

	// �c��̗v�f�������B
	// �����͔ėp���߁B
	for (; i < length; i++)
	{
		sum += a[i];
		squared_sum += a[i] * a[i];
	}

	double average = (double)sum / length;
	double squared_average = (double)squared_sum / length;

	return squared_average - (average * average);
}

int main(void)
{
	int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	int length = sizeof(a) / sizeof(a[0]);

	double result = dispersion_general(a, length);
	printf("dispersion_general: %lf\n", result);

	result = dispersion(a, length);
	printf("dispersion        : %lf\n", result);

	return 0;
}
