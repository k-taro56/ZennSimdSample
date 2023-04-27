// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-min-of-max-of-array

#include <stdio.h>
#include <limits.h>
#include <intrin.h>

// 汎用命令を使った、配列 a の中から最小値を求める関数。
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

// SIMD 命令を使った、配列 a の中から最小値を求める関数。
int min_of(const int a[], int length)
{
	int i = 0;

	// 最小値を最大の整数で初期化。
	__m256i min_value256 = _mm256_set1_epi32(INT_MAX);

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// 最小値をスカラー値に変換。
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

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < length; i++)
	{
		if (a[i] < min_value)
		{
			min_value = a[i];
		}
	}

	// 最小値を返す。
	return min_value;
}

// より最適化された、配列 a の中から最小値を求める関数。
int min_of_fast(const int a[], int length)
{
	int i;
	int min_value;
	__m256i min_value256;

	// 配列の要素数が 8 未満の場合は、汎用命令を使う。
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

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// 残りの要素を処理。
	// 配列の一部を重複して探索することになるが、結果に影響はない。
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// 配列の末尾から 8 要素分手前の位置からデータを読み込む。
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[length - (sizeof(__m256i) / sizeof(int))]));
		min_value256 = _mm256_min_epi32(min_value256, a256);
	}

	// 最小値をスカラー値に変換。
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

	// 最小値を返す。
	return min_value;
}

// 汎用命令を使った、配列 a の中から最大値を求める関数。
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

// SIMD 命令を使った、配列 a の中から最大値を求める関数。
int max_of(const int a[], int length)
{
	int i = 0;

	// 最大値を最小の整数で初期化。
	__m256i max_value256 = _mm256_set1_epi32(INT_MIN);

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// 最大値をスカラー値に変換。
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

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < length; i++)
	{
		if (a[i] > max_value)
		{
			max_value = a[i];
		}
	}

	// 最大値を返す。
	return max_value;
}

// より最適化された、配列 a の中から最大値を求める関数。
int max_of_fast(const int a[], int length)
{
	int i;
	int max_value;
	__m256i max_value256;

	// 配列の要素数が 8 未満の場合は、汎用命令を使う。
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

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*) & a[i]);
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// 残りの要素を処理。
	// 配列の一部を重複して探索することになるが、結果に影響はない。
	if (length % (sizeof(__m256i) / sizeof(int)) != 0)
	{
		// 配列の末尾から 8 要素分手前の位置からデータを読み込む。
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[length - (sizeof(__m256i) / sizeof(int))]));
		max_value256 = _mm256_max_epi32(max_value256, a256);
	}

	// 最大値をスカラー値に変換。
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

	// 最大値を返す。
	return max_value;
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0 };
	int length = sizeof(a) / sizeof(int);
	int min_value;
	int max_value;

	// 配列 a の中から最小値を求める。
	min_value = min_of_general(a, length);
	printf("min_value_general = %d\n", min_value);

	// 配列 a の中から最大値を求める。
	max_value = max_of_general(a, length);
	printf("max_value_general = %d\n", max_value);

	// 配列 a の中から最小値を求める。
	min_value = min_of(a, length);
	printf("min_value = %d\n", min_value);

	// 配列 a の中から最大値を求める。
	max_value = max_of(a, length);
	printf("max_value = %d\n", max_value);

	// 配列 a の中から最小値を求める。
	min_value = min_of_fast(a, length);
	printf("min_value_fast = %d\n", min_value);

	// 配列 a の中から最大値を求める。
	max_value = max_of_fast(a, length);
	printf("max_value_fast = %d\n", max_value);

	return 0;
}
