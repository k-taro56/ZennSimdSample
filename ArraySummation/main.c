// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k-taro56/articles/simd-array-summation

#include <stdio.h>
#include <intrin.h>

// �ėp���߂��g�����A�z�� a �̑S�v�f�̘a�����߂�֐��B
int sum_general(const int a[], int length)
{
	int sum = 0;

	for (int i = 0; i < length; i++)
	{
		sum += a[i];
	}

	return sum;
}

// SIMD ���߂��g�����A�z�� a �̑S�v�f�̘a�����߂�֐��B
int sum(const int a[], int length)
{
	int i = 0;
	// ���v�l�� 0 �ŏ������B
	__m256i sum256 = _mm256_setzero_si256();

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		sum256 = _mm256_add_epi32(sum256, a256);
	}

	// ���v�l���X�J���[�l�ɕϊ��B
	// �ėp���߂ł��������x�͕ς��Ȃ��B
	// https://stackoverflow.com/questions/42000693/why-my-avx2-horizontal-addition-function-is-not-faster-than-non-simd-addition
	__m256i sum256_permute = _mm256_permute2x128_si256(sum256, sum256, 1);
	__m256i result256 = _mm256_hadd_epi32(sum256, sum256_permute);
	result256 = _mm256_hadd_epi32(result256, result256);
	result256 = _mm256_hadd_epi32(result256, result256);
	int sum = _mm256_extract_epi32(result256, 0);

	// �c��̗v�f�������B
	// �����͔ėp���߁B
	for (; i < length; i++)
	{
		sum += a[i];
	}

	return sum;
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	int length = sizeof(a) / sizeof(int);

	int result = sum_general(a, length);
	printf("sum_general = %d\n", result);

	result = sum(a, length);
	printf("sum         = %d\n", result);
}
