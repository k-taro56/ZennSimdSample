// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-scalar-multiplication

#include <stdio.h>
#include <intrin.h>

// 汎用命令を使った、行列のスカラー倍を計算する関数。
void scalar_multiplication_general(int* a, int row, int column, int scalar)
{
	for (int i = 0; i < row * column; i++)
	{
		a[i] *= scalar;
	}
}

// SIMD 命令を使った、行列のスカラー倍を計算する関数。
void scalar_multiplication(int* a, int row, int column, int scalar)
{
	int i = 0;

	__m256i scalar256 = _mm256_set1_epi32(scalar);

	// 8 要素ずつ計算する。
	for (; i < row * column; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i product256 = _mm256_mullo_epi32(a256, scalar256);
		_mm256_storeu_si256((__m256i*)(&a[i]), product256);
	}

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < row * column; i++)
	{
		a[i] *= scalar;
	}
}

#define ROW 4
#define COLUMN 4

int main()
{
	int a[ROW][COLUMN] =
	{
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 }
	};

	scalar_multiplication_general((int*)a, 4, 4, 3);
	printf("scalar_multiplication_general:\n");

	for (int row = 0; row < ROW; row++)
	{
		for (int column = 0; column < COLUMN; column++)
		{
			printf("%3d", a[row][column]);
		}
		printf("\n");
	}

	int b[ROW][COLUMN] =
	{
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 }
	};

	scalar_multiplication((int*)b, 4, 4, 3);
	printf("\nscalar_multiplication:\n");

	for (int row = 0; row < ROW; row++)
	{
		for (int column = 0; column < COLUMN; column++)
		{
			printf("%3d", b[row][column]);
		}
		printf("\n");
	}

	return 0;
}
