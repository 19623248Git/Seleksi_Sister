#include "stdio.h"

#define MODULUS 998244353
#define PRIM_ROOT 15311432
#define ROOT_PW (1 << 23)
#define ARRAY_SIZE 4194304 

unsigned int input1[ARRAY_SIZE] = {0};
unsigned int input2[ARRAY_SIZE] = {0};
char input_s1[ARRAY_SIZE] = {0};
char input_s2[ARRAY_SIZE] = {0};

static inline unsigned int add(unsigned int a, unsigned int b) {
	unsigned int carry;
	add_loop:
	carry = (a & b) << 1;
	a = a ^ b;
	b = carry;
	if (b != 0) goto add_loop;
	return a;
}

static inline unsigned int sub(unsigned int a, unsigned int b) {
    	return add(a, add(~b, 1));
}

static inline unsigned long long add_ll(unsigned long long a, unsigned long long b) {
	unsigned long long carry;
	add_loop_ll:
	carry = (a & b) << 1;
	a = a ^ b;
	b = carry;
	if (b != 0) goto add_loop_ll;
	return a;
}

static inline unsigned long long sub_ll(unsigned long long a, unsigned long long b) {
	return add_ll(a, add_ll(~b, 1));
}

static inline unsigned long long mult_ll(unsigned long long a, unsigned long long b) {
	unsigned long long result = 0;
	mult_loop_ll:
	if (b == 0) return result;
	if (b & 1) result = add_ll(result, a);
	a <<= 1;
	b >>= 1;
	goto mult_loop_ll;
}

static inline unsigned long long mod_ll(unsigned long long a, unsigned long long b) {
	if (b == 0 || a < b) {
		return a;
	}
	unsigned long long temp_b = b;
	mod_align_loop_ll:
	if ((temp_b <= a) && ((temp_b << 1) > temp_b)) {
		temp_b <<= 1;
		goto mod_align_loop_ll;
	}
	mod_sub_loop_ll:
	if (a >= temp_b) {
		a = sub_ll(a, temp_b);
	}
	if (temp_b == b) {
		return a;
	}
	temp_b >>= 1;
	goto mod_sub_loop_ll;
}

static inline unsigned int divide_by_10(unsigned long long n) {
        unsigned long long temp = mult_ll((unsigned long long)n,0xCCCCCCCD);
        return (unsigned int)(temp >> 35);
}

static inline unsigned long long binexp_ll(unsigned long long a, unsigned long long b, unsigned long long m) {
        unsigned long long result = 1;
        a = mod_ll(a, m);
        binexp_loop:
        if (b == 0) return result;
        if (b & 1) result = mod_ll(mult_ll(result, a), m);
        a = mod_ll(mult_ll(a, a), m);
        b >>= 1;
        goto binexp_loop;
}

static inline unsigned long long inverse_ll(unsigned long long n, unsigned long long mod) {
    	return binexp_ll(n, sub_ll(mod, 2), mod);
}

unsigned int my_strlen(char* str) {
	unsigned int i = 0;
	lencheck:
	if (str[i] != '\0') {
		i = add(i, 1);
		goto lencheck;
	}
	return i;
}

void fft(unsigned int* arr, int invert, unsigned int n) {

	const unsigned int modulo = MODULUS;
	const unsigned int root = PRIM_ROOT;
	const unsigned int root_1 = (unsigned int)inverse_ll(root, modulo);
	const unsigned int root_pw = ROOT_PW;

	unsigned int i = 1, j = 0;
	bitReversePerm:
	if (i < n) {
		unsigned int bit = n >> 1;
		nestedLoopPerm:
		if (j & bit) {
		j ^= bit;
		bit >>= 1;
		goto nestedLoopPerm;
		}
		j ^= bit;
		if (i < j) {
		unsigned int swap_val = arr[i];
		arr[i] = arr[j];
		arr[j] = swap_val;
		}
		i = add(i, 1);
		goto bitReversePerm;
	}

	unsigned int len = 2;
	CTFFT:
	if (len <= n) {
		unsigned long long wlen = invert ? root_1 : root;
		unsigned int i1 = len;
		CTTFT1:
		if (i1 < root_pw) {
			wlen = mod_ll(mult_ll(wlen, wlen), modulo);
			i1 <<= 1;
			goto CTTFT1;
		}

		unsigned int i2 = 0;
		CTTFT2:
		if (i2 < n) {
			unsigned long long w = 1;
			unsigned int j_inner = 0;
			CTTFT21:
			if (j_inner < (len >> 1)) {
				unsigned int idx1 = add(i2, j_inner);
				unsigned int idx2 = add(idx1, len >> 1);
				unsigned long long u = arr[idx1];
				unsigned long long v = mod_ll(mult_ll((unsigned long long)arr[idx2], w), modulo);

				arr[idx1] = (unsigned int)mod_ll(add_ll(u, v), modulo);
				arr[idx2] = (unsigned int)mod_ll(sub_ll(add_ll(u, modulo), v), modulo);
				
				w = mod_ll(mult_ll(w, wlen), modulo);
				j_inner = add(j_inner, 1);
				goto CTTFT21;
			}
			i2 = add(i2, len);
			goto CTTFT2;
		}
		len <<= 1;
		goto CTFFT;
	}

	if (invert) {
		unsigned long long n_1 = inverse_ll(n, modulo);
		unsigned int i3 = 0;
		inversescale:
		if (i3 < n) {
			arr[i3] = (unsigned int)mod_ll(mult_ll((unsigned long long)arr[i3], n_1), modulo);
			i3 = add(i3, 1);
			goto inversescale;
		}
	}
}

int main() {
	//     printf("Enter first number:\n");
	scanf("%s", input_s1);
	//     printf("Enter second number:\n");
	scanf("%s", input_s2);

	unsigned int n1 = my_strlen(input_s1);
	unsigned int n2 = my_strlen(input_s2);

	unsigned int i = 0;
	LoopPush1:
	if (i < n1) {
		input1[i] = input_s1[sub(sub(n1, 1), i)] - '0';
		i = add(i, 1);
		goto LoopPush1;
	}
	
	i = 0;
	LoopPush2:
	if (i < n2) {
		input2[i] = input_s2[sub(sub(n2, 1), i)] - '0';
		i = add(i, 1);
		goto LoopPush2;
	}

	unsigned int n = 1;
	loopn:
	if (n < add(n1, n2)) {
		n <<= 1;
		goto loopn;
	}
	
	fft(input1, 0, n);
	fft(input2, 0, n);

	const unsigned int modulo = MODULUS;
	unsigned int i0 = 0;
	pointwiseloop:
	if (i0 < n) {
		input1[i0] = (unsigned int)mod_ll(mult_ll((unsigned long long)input1[i0], (unsigned long long)input2[i0]), modulo);
		i0 = add(i0, 1);
		goto pointwiseloop;
	}

	fft(input1, 1, n);
	
	unsigned long long carry = 0;
	unsigned int i1 = 0;
	carryhandle:
	if (i1 < n) {
		unsigned long long current_val = add_ll((unsigned long long)input1[i1], carry);
		input1[i1] = (unsigned int)mod_ll(current_val, 10);
		carry = divide_by_10(current_val);
		i1 = add(i1, 1);
		goto carryhandle;
	}
	
	unsigned int final_idx = n;
	final_carry_loop:
	if(carry > 0){
		input1[final_idx] = (unsigned int)mod_ll(carry, 10);
		carry = divide_by_10(carry);
		final_idx = add(final_idx, 1);
		goto final_carry_loop;
	}

	int first_digit = sub(final_idx, 1);
	if (final_idx == 0) first_digit = 0;

	findmostsignificant:
	if (first_digit > 0 && input1[first_digit] == 0) {
		first_digit = sub(first_digit, 1);
		goto findmostsignificant;
	}
	
	unsigned int i2 = first_digit;
	printres:
	if (i2 < ARRAY_SIZE) {
		printf("%d", input1[i2]);
		if (i2 == 0) goto EndPrint;
		i2 = sub(i2, 1);
		goto printres;
	}
	EndPrint:;

	printf("\n");
	return 0;
}