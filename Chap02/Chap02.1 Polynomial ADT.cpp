/* Chap02.1 Polynomial ADT */
#include <stdio.h> 
#include <stdlib.h>
#define MAX_TERM 4

typedef struct
{
	float coef; // coefficient
	int expon;  // exponent
} polynomial;

/* specification operation */ 
void Zero(polynomial arr[]);
int IsZero(polynomial *poly);
float Coef(polynomial p[], int);
int Lead_Exp(polynomial p[]);
void Attach(polynomial p[], float, int);
void Remove(polynomial *p, int);
void SingleMult(polynomial *p, float, int);


int main(void)
{
	polynomial a[MAX_TERM] = {{0, 0}, {4, 1}, {7, 2}, {0, 3}}; //a = 0 + 4x + 7x^2 + 0
	polynomial b[MAX_TERM] = {{2, 0}, {0, 1}, {3, 2}, {5, 3}}; //b = 2 + 0  + 3x^2 + 5x^3
	polynomial p[MAX_TERM] = {{0, 0}, {0, 0}, {4, 2}, {2, 3}}; //b = 0 + 0  + 4x^2 + 2x^3
	polynomial zero[MAX_TERM]; 
	int i, is_zero, largest_degree;
	float coeff;
	
	Zero(zero); // let zero be zero polynomial
	is_zero = IsZero(zero);
	printf("is zero a zero polynomial (1,Yes / 0,NO)? %d\n", is_zero);
	
	coeff = Coef(a, 1); // find coefficient of a when degree = 1
	printf("coefficient of degree 1 of polynomial a : %2.1f\n", coeff);
	
	largest_degree = Lead_Exp(a); //find largest degree of a
	printf("largest degree of a is : %d\n", largest_degree);
	
	printf("term 1 in p is : coef = %2.1f, expon = %d\n", p[1].coef, p[1].expon);
	Attach(p, 7, 1); // add 7x to p
	printf("term 1 in p is : coef = %2.1f, expon = %d\n", p[1].coef, p[1].expon);
	
	printf("term 2 in a is : coef = %2.1f, expon = %d\n", a[2].coef, a[2].expon);
	Remove(a, 2); // remove 7x^2 in a
	printf("term 2 in a is : coef = %2.1f, expon = %d\n", a[2].coef, a[2].expon);
	
	
	printf("p = ");
	for (i=0; i<MAX_TERM; i++)
		printf("<%2.1f, %d> ", p[i].coef, p[i].expon); // p = 0 + 7x^1  + 4x^2 + 2x^3
	printf("\n");
	SingleMult(p, 2, 1); // product 2x
	printf("p = ");
	for (i=0; i<MAX_TERM; i++)
		printf("<%2.1f, %d> ", p[i].coef, p[i].expon); // p = 0 + 14x^2 + 8x^3 + 4x^4
	printf("\n");
	
	
	system("pause");
	return 0;
}

void Zero(polynomial arr[]) // initialize a polynomial to zero
{
	int i;
	for (i=0; i<MAX_TERM; i++)
	{
		(arr+i)->coef = 0;
		(arr+i)->expon = 0;
	}
}
int IsZero(polynomial *poly)
{
	int i;
	for (i=0; i<MAX_TERM; i++)
	{
		if ((poly+i)->coef == 0)
			continue;
		else
			return 0; // not zero polynomial
	}
	return 1;
}
float Coef(polynomial p[], int n) // find particular coefficient
{
	int i;
	for (i=0; i<MAX_TERM; i++)
	{
		if (p[i].expon == n)
			return p[i].coef;
		else
			continue;
	}
	return 0.0;
}
int Lead_Exp(polynomial p[]) // fina largest exponent
{
	int i, index=0;
	for (i=0; i<MAX_TERM; i++)
	{
		if (p[i].coef != 0)
			index = i;
	}
	return p[index].expon;
}
void Attach(polynomial p[], float c, int n) // add particular some term with degree n
{
	p[n].expon = n;
	p[n].coef = c;
}
void Remove(polynomial *p, int n) // remove particular some term with degree n
{
	(p+n)->coef = 0;
	(p+n)->expon = 0;
}
void SingleMult(polynomial *p, float c, int n)
{
	int i;
	for (i=0; i<MAX_TERM; i++)
	{
		(p+i)->coef = (p+i)->coef * c;
		(p+i)->expon = (p+i)->expon + n;
	}
}
