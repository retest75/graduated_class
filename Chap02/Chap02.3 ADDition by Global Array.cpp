/* Chap02.3 ADDition by Global Array */
#include <stdio.h> 
#include <stdlib.h>
#define SIZE 20


typedef struct
{
	int coef;
	int expon;
} polynomial;

/* A(x) = 2x^1000 + 1, B(x) = x^4 + 10x^3 + 3x^2 + 1 */
polynomial term[SIZE] = {{2, 1000}, {1, 0}, {1, 4}, {10, 3}, {3, 2}, {1, 0}};
int startA = 0, finishA = 1;
int startB = 2, finishB = 5;
int avail = 6;
int startD = 6, finishD;

void padd(int , int, int, int);
int compare(int, int);
void attach(float, int);

int main(void)
{
	int i;
	
	/* show polynomial A(x) = 2x^1000 + 1 */
	printf("A(x) = ");
	for (i=startA; i<=finishA; i++)
		printf("<%d,%d> ", term[i].coef, term[i].expon);
	printf("\n");
	
	/* show polynomial B(x) = x^4 + 10x^3 + 3x^2 + 1 */
	printf("B(x) = ");
	for (i=startB; i<=finishB; i++)
		printf("<%d,%d> ", term[i].coef, term[i].expon);
	printf("\n");
	
	/* D(x) = A(x) + B(x) */
	padd(startA, finishA, startB, finishB);
	finishD = avail - 1;  // finish position of D
	
	/* show polynomial D(x) = 2x^1000 + x^4 + 10x^3 + 3x^2 + 1 + 1 */
	printf("D(x) = ");
	for (i=startD; i<=finishD; i++)
		printf("<%d, %d> ", term[i].coef, term[i].expon);
	printf("\n");
	
	
}

void padd(int sA, int fA, int sB, int fB)
{
	float coefficient;
	//startD = avail; // start position of D(x)
	
	while (sA<=fA && sB<=fB)
	{
		switch (compare(term[sA].expon, term[sB].expon))
		{
			case -1:   // largest expon of A < largest expon of B
				attach(term[sB].coef, term[sB].expon);
				sB++;
				break;
			case 0:   // largest expon of A = largest expon of B
				coefficient = term[sA].coef + term[sB].coef;
				if (coefficient)
				{
					attach(coefficient, term[sA].expon);
					sA++;
					sB++;
				}
				else
				{
					attach(0.0, term[sA].expon);
					sA++;
					sB++;
				}
				break;
			case 1:   // largest expon of A > largest expon of B
				attach(term[sA].coef, term[sA].expon);
				sA++;
				break;
		}
	}
	
	for (; sA<=fA; sA++)
		attach(term[sA].coef, term[sA].expon);
	for (; sB<=fB; sB++)
		attach(term[sB].coef, term[sB].expon);
}

int compare(int expon_A, int expon_B)
{
	if (expon_A < expon_B)
		return -1;
	else if (expon_A == expon_B)
		return 0;
	else
		return 1;
}
void attach(float coef, int expon)
{
	term[avail].coef = coef;
	term[avail++].expon = expon;
}
