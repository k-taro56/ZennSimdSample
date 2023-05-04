// MIT License
// Refer to LICENSE.txt for more information.

// https://zenn.dev/k_taro56/articles/simd-vector-dot-product

#include <stdio.h>
#include <intrin.h>

// 汎用命令を使った、ベクトルの内積を求める関数。
int dot_product_general(const int a[], const int b[], int length)
{
	int dot_product = 0;

	for (int i = 0; i < length; i++)
	{
		dot_product += a[i] * b[i];
	}

	return dot_product;
}

// SIMD 命令を使った、ベクトルの内積を求める関数。
int dot_product(const int a[], const int b[], int length)
{
	int i = 0;

	// 合計を 0 で初期化。
	__m256i dot_product256 = _mm256_setzero_si256();

	// 各要素を 8 個ずつ処理。
	for (; i + 7 < length; i += 8)
	{
		__m256i a256 = _mm256_loadu_si256((__m256i*)(&a[i]));
		__m256i b256 = _mm256_loadu_si256((__m256i*)(&b[i]));

		__m256i product256 = _mm256_mullo_epi32(a256, b256);
		dot_product256 = _mm256_add_epi32(dot_product256, product256);
	}

	// スカラー値に変換。
	// 汎用命令でも実効速度は変わらない。
	// https://stackoverflow.com/questions/42000693/why-my-avx2-horizontal-addition-function-is-not-faster-than-non-simd-addition
	__m256i dot_product256_permute = _mm256_permute2x128_si256(dot_product256, dot_product256, 1);
	__m256i sum256 = _mm256_hadd_epi32(dot_product256, dot_product256_permute);
	sum256 = _mm256_hadd_epi32(sum256, sum256);
	sum256 = _mm256_hadd_epi32(sum256, sum256);
	int dot_product = _mm256_extract_epi32(sum256, 0);

	// 残りの要素を処理。
	// ここは汎用命令。
	for (; i < length; i++)
	{
		dot_product += a[i] * b[i];
	}

	return dot_product;
}

int main(void)
{
	int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int b[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int length = sizeof(a) / sizeof(int);

	int dot_product_result = dot_product_general(a, b, length);
	printf("dot_product_general: %d\n", dot_product_result);

	dot_product_result = dot_product(a, b, length);
	printf("dot_product        : %d\n", dot_product_result);

	return 0;
}
