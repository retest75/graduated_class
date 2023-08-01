/* Chap04.4 Represent Stack */ 
#include <stdio.h>
#include <stdlib.h>
#define MAX_STACK_SIZE 10

/* define a stack element */
typedef struct
{
	int key;
} element;

/* define linked stack node and its pointer */
typedef struct LinkedStack_node LinkedStack_node;
typedef LinkedStack_node *LinkedStack_pointer;
struct LinkedStack_node
{
	element data;
	LinkedStack_pointer link;
};

/* declare global variable and prototype */
LinkedStack_pointer stack[MAX_STACK_SIZE];  // linked stack
LinkedStack_node *top = NULL;             // point to top node
int size = -1;                              // current size of stack
void push(LinkedStack_pointer*, element);

int main(void)
{
	/* declare variable */
	LinkedStack_pointer node;
	element a = {84};
	element b = {11};
	
	/* set up initial value */
	node = (LinkedStack_pointer) malloc(sizeof(LinkedStack_node));
	node->data = a;
	node->link = NULL;
	top = node;
	stack[0] = node;
	++size;
	
	printf("%d\n", stack[size]->data.key); // 84
	printf("%p\n", stack[size]->link);     // 0000000000000000
	printf("%p\n", top);
	printf("%p\n", node);
	
	push(&top, b);
	printf("%p\n", top);
	printf("%d\n", size);
	printf("%d\n", stack[size]->data.key);
	printf("%p\n", stack[size]->link);
	//printf("%p\n", stack[size]);
	//printf("%p\n", top);
	//printf("%d\n", size);
	
	free(stack[size]);
	free(node);
	
	system("pause");
	return 0;
}
void push(LinkedStack_pointer* top_ptr, element data) // top_ptr is a pointer , pointing to address of top 
{
	LinkedStack_pointer temp;
	
	if (size == (MAX_STACK_SIZE-1))
		printf("the stack is full !\n");
	else
	{
		temp = (LinkedStack_pointer) malloc(sizeof(LinkedStack_node));
		temp->data = data;
		temp->link = *top_ptr; // *top_ptr is content of pointer top_ptr , i.e top
		*top_ptr = temp;
		++size;
	}
}


