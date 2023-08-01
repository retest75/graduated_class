/* Chap03.5 Evaluation of Postfix Expressions */
#include <stdio.h>
#include <stdlib.h>
#define MAX_STACK_SIZE 100

typedef enum
{
	lparan,	rparan,                       // '(',  ')'
	plus,	minus,	times,	divide,	mod,  // '+', '-', '*', '/'
	eos,                                  // '\0'
	operand
} precedence;

/* prototype */
precedence get_token(char *, int *);
void push(int *, int);
int pop(int *);

/* global stack */
int stack[MAX_STACK_SIZE]; // save operand of postfix expressions
char expression[MAX_STACK_SIZE] = "62/3-42*+"; // save input string

int main(void)
{
	char symbol;         // read symbol from expression
	precedence token; 
	int op1, op2;        // two operands
	int n=0;             //current position of expression
	int top=-1;
	
	token = get_token(&symbol, &n);
	while(token != eos)
	{
		if (token == operand)
			push(&top, symbol-'0');
		else
		{
			op1 = pop(&top);
			op2 = pop(&top);
			switch (token)
			{
				case plus:
					push(&top, op2+op1);
					break;
				case minus:
					push(&top, op2-op1);
					break;
				case times:
					push(&top, op2*op1);
					break;
				case divide:
					push(&top, op2/op1);
					break;
				case mod:
					push(&top, op2%op1);
					break;
			}
		}
		printf("top value\n");
		printf("%2d%5d\n", top, stack[top]);
		token = get_token(&symbol, &n);
	}
	
	system("pause");
	return 0;
}

