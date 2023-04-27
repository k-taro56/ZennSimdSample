// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-index-of-array

#include <stdio.h>
#include <intrin.h>

// 32 ビット整数の 8 個の要素を持つベクトルの中から、最初に 0 以外の要素が見つかったインデックスを求める関数。
unsigned long find_first_non_zero_index_epi32(__m256i a)
{
	unsigned long index;
	__m256 floating_point_a = _mm256_castsi256_ps(a);
	int mask = _mm256_movemask_ps(floating_point_a);
	_BitScanForward(&index, mask);
	return index;
}

// 汎用命令を使った、配列 a の中から key と等しい要素のインデックスを求める関数。
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

// SIMD 命令を使った、配列 a の中から key と等しい要素のインデックスを求める関数。
int index_of(const int a[], int length, int key)
{
	if (length < 0)
	{
		return -1;
	}

	int i = 0;

	__m256i key256 = _mm256_set1_epi32(key);

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 個の要素の中に key と等しい要素があるかどうかを判定。
		if (!_mm256_testz_si256(equals256, equals256))
		{
			return i + find_first_non_zero_index_epi32(equals256);
		}
	}

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < length; i++)
	{
		if (key == a[i])
		{
			return i;
		}
	}

	return -1;
}

// より最適化された、配列 a の中から key と等しい要素のインデックスを求める関数。
int index_of_fast(const int a[], int length, int key)
{
	if (length < 0)
	{
		return -1;
	}

	int i;

	// 配列の要素数が 8 未満の場合は、汎用命令を使う。
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

	// 各要素を 8 個ずつ処理。
	for (i = 0; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 個の要素の中に key と等しい要素があるかどうかを判定。
		if (!_mm256_testz_si256(equals256, equals256))
		{
			return i + find_first_non_zero_index_epi32(equals256);
		}
	}

	// 残りの要素を処理。
	// 配列の一部を重複して探索することになるが、結果に影響はない。
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// 配列の末尾から 8 要素分手前の位置からデータを読み込む。
		i = length - (sizeof(__m256i) / sizeof(int));
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i equals256 = _mm256_cmpeq_epi32(a256, key256);

		// 8 個の要素の中に key と等しい要素があるかどうかを判定。
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
