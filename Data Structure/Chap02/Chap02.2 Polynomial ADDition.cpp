/* Chap02.2 Polynomial ADDition */
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

int compare(int, int);



int main(void)
{
	polynomial a[MAX_TERM] = {{0, 0}, {4, 1}, {7, 2}, {0, 3}}; //a = 0 + 4x + 7x^2 + 0
	polynomial b[MAX_TERM] = {{2, 0}, {0, 1}, {3, 2}, {5, 3}}; //b = 2 + 0  + 3x^2 + 5x^3
	polynomial d[MAX_TERM];                                    //d = a + b = 2 + 4x + 10x^2 + 5x^3
	int i, sum=0;
	float coeff;
	
	Zero(d); // let d be zero polynomial
	
	while (! IsZero(a) || ! IsZero(b))
	{
		switch (compare(Lead_Exp(a), Lead_Exp(b)))
		{
			case -1:
				Attach(d, Coef(b, Lead_Exp(b)), Lead_Exp(b));
				Remove(b, Lead_Exp(b));
				break;
			case 0:
				sum = Coef(a, Lead_Exp(a)) + Coef(b, Lead_Exp(b));
				Attach(d, sum, Lead_Exp(a));
				Remove(a, Lead_Exp(a));
				Remove(b, Lead_Exp(b));
				break;
			case 1:
				Attach(d, Coef(a, Lead_Exp(a)), Lead_Exp(a));
				Remove(a, Lead_Exp(a));
				break;
		}
	}
	
	/* show polynomial d */
	for (i=0; i<MAX_TERM; i++)
		printf("<%2.1f, %d> ", d[i].coef, d[i].expon);
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
int compare(int a, int b)
{
	if (a < b)
		return -1;
	else if (a == b)
		return 0;
	else
		return 1;
}
