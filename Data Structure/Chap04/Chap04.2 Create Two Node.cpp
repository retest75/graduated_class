/* Chap04.2 Create Two Node */ 
#include <stdio.h>
#include <stdlib.h>
#define IS_EMPTY(ptr) (!(ptr))

typedef struct list_node list_node;
typedef list_node *list_pointer;
struct list_node
{
	int data;
	list_pointer link;
};

/* declare NULL and prototype */
list_pointer ptr=NULL;
list_pointer create_two_node(void);
void print_list(list_pointer);

int main(void)
{
	list_pointer first;
	
	first = create_two_node(); /* create a list having two node */
	printf("IS first empty? ");
	if (IS_EMPTY(first))
		printf("Yes\n");
	else
	{
		printf("No\n");
		printf("first node data:  %d\n", first->data);
		printf("first node link: %p\n", first->link);
		printf("second node data: %d\n", first->link->data);
		printf("second node link: %p\n", first->link->link);
	}
	printf("print all data of node: ");
	print_list(first);
	
	free(first->link); /* free second node */
	free(first);       /* free first node */
	
	system("pause");
	return 0;
}
list_pointer create_two_node(void)
{
	list_pointer first, second;
	
	first = (list_pointer) malloc(sizeof(list_node));
	second = (list_pointer) malloc(sizeof(list_node));
	first->data = 10;
	first->link = second;
	second->data = 20;
	second->link = NULL;
	printf("second node address: %p\n", second);
	
	return first;
}
void print_list(list_pointer ptr)
{
	for (; ptr; ptr=ptr->link) // ptr point to next node
		printf("%d ", ptr->data);
	printf("\n");
}

