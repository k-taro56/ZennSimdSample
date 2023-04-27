// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-index-of-array

#include <stdio.h>
#include <intrin.h>

// 32 �r�b�g������ 8 �̗v�f�����x�N�g���̒�����A�ŏ��� 0 �ȊO�̗v�f�����������C���f�b�N�X�����߂�֐��B
unsigned long find_first_non_zero_index_epi32(__m256i a)
{
	unsigned long index;
	__m256 floating_point_a = _mm256_castsi256_ps(a);
	int mask = _mm256_movemask_ps(floating_point_a);
	_BitScanForward(&index, mask);
	return index;
}

// �ėp���߂��g�����A�z�� a �̒����� key �Ɠ������v�f�̃C���f�b�N�X�����߂�֐��B
int index_of_general(const int a[], int length, int key)
{
	for (int i = 0; i < length; i++)
	{
		if (key == a[i])
		{
			return i;
		}
	}

	return -1;
}

// SIMD ���߂��g�����A�z�� a �̒����� key �Ɠ������v�f�̃C���f�b�N�X�����߂�֐��B
int index_of(const int a[], int length, int key)
{
	if (length < 0)
	{
		return -1;
	}

	int i = 0;

	__m256i key256 = _mm256_set1_epi32(key);

	// �e�v�f�� 8 �������B
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 �̗v�f�̒��� key �Ɠ������v�f�����邩�ǂ����𔻒�B
		if (!_mm256_testz_si256(equals256, equals256))
		{
			return i + find_first_non_zero_index_epi32(equals256);
		}
	}

	// �c��̗v�f�������B
	// �����͔ėp���߁B
	for (; i < length; i++)
	{
		if (key == a[i])
		{
			return i;
		}
	}

	return -1;
}

// ���œK�����ꂽ�A�z�� a �̒����� key �Ɠ������v�f�̃C���f�b�N�X�����߂�֐��B
int index_of_fast(const int a[], int length, int key)
{
	if (length < 0)
	{
		return -1;
	}

	int i;

	// �z��̗v�f���� 8 �����̏ꍇ�́A�ėp���߂��g���B
	if (length < 8)
	{
		for (i = 0; i < length; i++)
		{
			if (key == a[i])
			{
				return i;
			}
		}

		return -1;
	}

	__m256i key256 = _mm256_set1_epi32(key);

	// �e�v�f�� 8 �������B
	for (i = 0; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 �̗v�f�̒��� key �Ɠ������v�f�����邩�ǂ����𔻒�B
		if (!_mm256_testz_si256(equals256, equals256))
		{
			return i + find_first_non_zero_index_epi32(equals256);
		}
	}

	// �c��̗v�f�������B
	// �z��̈ꕔ���d�����ĒT�����邱�ƂɂȂ邪�A���ʂɉe���͂Ȃ��B
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// �z��̖������� 8 �v�f����O�̈ʒu����f�[�^��ǂݍ��ށB
		i = length - (sizeof(__m256i) / sizeof(int));
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 �̗v�f�̒��� key �Ɠ������v�f�����邩�ǂ����𔻒�B
		if (!_mm256_testz_si256(equals256, equals256))
		{
			return i + find_first_non_zero_index_epi32(equals256);
		}
	}

	return -1;
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0 };
	int length = sizeof(a) / sizeof(int);
	int key = 0;
	int index;

	index = index_of_general(a, length, key);
	printf("index_of_general: key = %d, index = %d\n", key, index);

	index = index_of(a, length, key);
	printf("index_of:         key = %d, index = %d\n", key, index);

	index = index_of_fast(a, length, key);
	printf("index_of_fast:    key = %d, index = %d\n", key, index);

	key = 4;

	index = index_of_general(a, length, key);
	printf("index_of_general: key = %d, index = %d\n", key, index);

	index = index_of(a, length, key);
	printf("index_of:         key = %d, index = %d\n", key, index);

	index = index_of_fast(a, length, key);
	printf("index_of_fast:    key = %d, index = %d\n", key, index);

	key = 12;

	index = index_of_general(a, length, key);
	printf("index_of_general: key = %d, index = %d\n", key, index);

	index = index_of(a, length, key);
	printf("index_of:         key = %d, index = %d\n", key, index);

	index = index_of_fast(a, length, key);
	printf("index_of_fast:    key = %d, index = %d\n", key, index);

	return 0;
}
