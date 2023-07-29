/* Chap03.1 Stack ADT */

#include <stdio.h> 
#include <stdlib.h>
#define MAX_STACK_SIZE 5 

typedef struct
{
	int key; // other fields
} element;

// create a global stack
element stack[MAX_STACK_SIZE] = {{84}, {11}, {20}};
int top = 2;

/* operation function */
int IsFull(int);
int IsEmpty(int);
void push(int *, element);
element pop(int *top);

int main(void)
{
	int *top_ptr = &top;
	element item;           // delete element
	element a = {900607};   // add element
	
	printf("is full ? ");
	if (IsFull(top))
		printf("Full !\n");
	else
		printf("Not Full !\n");
		
	printf("is empty ? ");
	if (IsEmpty(top))
		printf("Empty !\n");
	else
		printf("Not Empty !\n");
	printf("----------\n");
	
	printf("before push, top = %d\n", top);
	push(top_ptr, a);
	printf("after push, top = %d\n", top);
	printf("just push: %d\n", stack[top].key);
	printf("----------\n");
	
	printf("before pop, top = %d\n", top);
	item = pop(top_ptr);
	printf("pop element is : %d\n", item.key);
	printf("after pop, top = %d\n", top);
	printf("current top element is : %d\n", stack[top].key);
	
	
	system("pause");
	return 0;
}
int IsFull(int top)
{
	if (top == MAX_STACK_SIZE-1)
		return 1;
	else
		return 0;	
}
int IsEmpty(int top)
{
	if (top == -1)
		return 1;
	else
		return 0;
}
void push(int *top, element item)
{
	if (*top == MAX_STACK_SIZE)
	{
		printf("th stack is full");
		return;
	}
	else
		stack[++*top] = item;
}
element pop(int *top) // return top element from the stack
{
	if (*top == -1)
	{
		printf("the stack is empty");
		element empty_stack = {0};
		return empty_stack;
	}
	else
		return stack[(*top)--];
}
