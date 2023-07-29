/* Chap03.5-1 stack operation */
#include <stdio.h>
#define MAX_STACK_SIZE 100



void push(int *top_ptr, int n)
{
	extern int stack[MAX_STACK_SIZE]; /* global stack */
	if (*top_ptr >= MAX_STACK_SIZE)
		printf("the stack is full");
	else
		stack[++*top_ptr] = n;
}

int pop(int *top_ptr)
{
	extern int stack[MAX_STACK_SIZE]; /* global stack */
	if (*top_ptr == -1)
	{
		printf("the stack is empty");
		return -1;
	}
	else
		return stack[(*top_ptr)--];
}

 
