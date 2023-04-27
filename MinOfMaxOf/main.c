// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-min-of-max-of-array

#include <stdio.h>
#include <limits.h>
#include <intrin.h>

// �ėp���߂��g�����A�z�� a �̒�����ŏ��l�����߂�֐��B
int min_of_general(const int a[], int length)
{
	int min_value = INT_MAX;

	for (int i = 0; i < length; i++)
	{
		if (a[i] < min_value)
		{
			min_value = a[i];
		}
	}

	return min_value;
}

// SIMD ���߂��g�����A�z�� a �̒�����ŏ��l�����߂�֐��B
int min_of(const int a[], int length)
{
	int i = 0;

	// �ŏ��l���ő�̐����ŏ������B
	__m256i min_value256 = _mm256_set1_epi32(INT_MAX);

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// �ŏ��l���X�J���[�l�ɕϊ��B
	int result[8];
	_mm256_storeu_si256((__m256i*)result, min_value256);

	int min_value = result[0];

	for (int j = 1; j < 8; j++)
	{
		if (result[j] < min_value)
		{
			min_value = result[j];
		}
	}

	// �c��̗v�f�������B
	// �����͔ėp���߁B
	for (; i < length; i++)
	{
		if (a[i] < min_value)
		{
			min_value = a[i];
		}
	}

	// �ŏ��l��Ԃ��B
	return min_value;
}

// ���œK�����ꂽ�A�z�� a �̒�����ŏ��l�����߂�֐��B
int min_of_fast(const int a[], int length)
{
	int i;
	int min_value;
	__m256i min_value256;

	// �z��̗v�f���� 8 �����̏ꍇ�́A�ėp���߂��g���B
	if (length < 8)
	{
		min_value = INT_MAX;

		for (i = 0; i < length; i++)
		{
			if (a[i] < min_value)
			{
				min_value = a[i];
			}
		}

		return min_value;
	}

	i = 8;
	min_value256 = _mm256_loadu_si256((__m256i*)(&a[0]));

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// �c��̗v�f�������B
	// �z��̈ꕔ���d�����ĒT�����邱�ƂɂȂ邪�A���ʂɉe���͂Ȃ��B
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// �z��̖������� 8 �v�f����O�̈ʒu����f�[�^��ǂݍ��ށB
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[length - (sizeof(__m256i) / sizeof(int))]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// �ŏ��l���X�J���[�l�ɕϊ��B
	int result[8];
	_mm256_storeu_si256((__m256i*)result, min_value256);

	min_value = result[0];

	for (int j = 1; j < 8; j++)
	{
		if (result[j] < min_value)
		{
			min_value = result[j];
		}
	}

	// �ŏ��l��Ԃ��B
	return min_value;
}

// �ėp���߂��g�����A�z�� a �̒�����ő�l�����߂�֐��B
int max_of_general(const int a[], int length)
{
	int max_value = INT_MIN;

	for (int i = 0; i < length; i++)
	{
		if (a[i] > max_value)
		{
			max_value = a[i];
		}
	}

	return max_value;
}

// SIMD ���߂��g�����A�z�� a �̒�����ő�l�����߂�֐��B
int max_of(const int a[], int length)
{
	int i = 0;

	// �ő�l���ŏ��̐����ŏ������B
	__m256i max_value256 = _mm256_set1_epi32(INT_MIN);

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// �ő�l���X�J���[�l�ɕϊ��B
	int result[8];
	_mm256_storeu_si256((__m256i*)result, max_value256);

	int max_value = result[0];

	for (int j = 1; j < 8; j++)
	{
		if (result[j] > max_value)
		{
			max_value = result[j];
		}
	}

	// �c��̗v�f�������B
	// �����͔ėp���߁B
	for (; i < length; i++)
	{
		if (a[i] > max_value)
		{
			max_value = a[i];
		}
	}

	// �ő�l��Ԃ��B
	return max_value;
}

// ���œK�����ꂽ�A�z�� a �̒�����ő�l�����߂�֐��B
int max_of_fast(const int a[], int length)
{
	int i;
	int max_value;
	__m256i max_value256;

	// �z��̗v�f���� 8 �����̏ꍇ�́A�ėp���߂��g���B
	if (length < 8)
	{
		max_value = INT_MIN;

		for (i = 0; i < length; i++)
		{
			if (a[i] > max_value)
			{
				max_value = a[i];
			}
		}

		return max_value;
	}

	i = 8;
	max_value256 = _mm256_loadu_si256((__m256i*)(&a[0]));

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*) & a[i]);
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// �c��̗v�f�������B
	// �z��̈ꕔ���d�����ĒT�����邱�ƂɂȂ邪�A���ʂɉe���͂Ȃ��B
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// �z��̖������� 8 �v�f����O�̈ʒu����f�[�^��ǂݍ��ށB
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[length - (sizeof(__m256i) / sizeof(int))]));
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// �ő�l���X�J���[�l�ɕϊ��B
	int result[8];
	_mm256_storeu_si256((__m256i*)result, max_value256);

	max_value = result[0];

	for (int j = 1; j < 8; j++)
	{
		if (result[j] > max_value)
		{
			max_value = result[j];
		}
	}

	// �ő�l��Ԃ��B
	return max_value;
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0 };
	int length = sizeof(a) / sizeof(int);
	int min_value;
	int max_value;

	// �z�� a �̒�����ŏ��l�����߂�B
	min_value = min_of_general(a, length);
	printf("min_value_general = %d\n", min_value);

	// �z�� a �̒�����ő�l�����߂�B
	max_value = max_of_general(a, length);
	printf("max_value_general = %d\n", max_value);

	// �z�� a �̒�����ŏ��l�����߂�B
	min_value = min_of(a, length);
	printf("min_value = %d\n", min_value);

	// �z�� a �̒�����ő�l�����߂�B
	max_value = max_of(a, length);
	printf("max_value = %d\n", max_value);

	// �z�� a �̒�����ŏ��l�����߂�B
	min_value = min_of_fast(a, length);
	printf("min_value_fast = %d\n", min_value);

	// �z�� a �̒�����ő�l�����߂�B
	max_value = max_of_fast(a, length);
	printf("max_value_fast = %d\n", max_value);

	return 0;
}
