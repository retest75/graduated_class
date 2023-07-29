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
	char symbol;       // read symbol from expression
	precedence token; 
	int op1, op2;      // two operands
	int n=0;           // current position of expression
	int top=-1;
	int r;
	
	token = get_token(&symbol, &n);
	while (token != eos)
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
	//printf("top = %d\n", top);
	//printf("62/3-42*+ = %d\n", stack[top]); // 6/2-3+4*2 = 8
	
	system("pause");
	return 0;
}
void push(int *top_ptr, int n)
{
	if (*top_ptr >= MAX_STACK_SIZE)
		printf("the stack is full");
	else
		stack[++*top_ptr] = n;
}

int pop(int *top_ptr)
{
	if (*top_ptr == -1)
	{
		printf("the stack is empty");
		return -1;
	}
	else
		return stack[(*top_ptr)--];
}
precedence get_token(char *symbl, int *n)
{
	*symbl = expression[(*n)++];
	switch (*symbl)
	{
		case '(':
			return lparan;
			break;
		case ')':
			return rparan;
			break;
		case '+':
			return plus;
			break;
		case '-':
			return minus;
			break;
		case '*':
			return times;
			break;
		case '/':
			return divide;
			break;
		case '%':
			return mod;
			break;
		case '\0':
			return eos;	
			break;
		default:
			return operand;
	}
}

