/* Chap04.3 Delete Node */ 
#include <stdio.h>
#include <stdlib.h>

typedef struct list_node list_node;
typedef list_node *list_pointer;
struct list_node
{
	int data;
	list_pointer link;
};

/* prototype */
void del(list_pointer *, list_pointer, list_pointer);
void print_list(list_pointer);
int main(void)
{
	/* construct list having three node */
	list_pointer first, second, third, node;
	list_pointer trail=NULL;
	first = (list_pointer) malloc(sizeof(list_node));
	second = (list_pointer) malloc(sizeof(list_node));
	third = (list_pointer) malloc(sizeof(list_node));
	third->link = NULL;
	third->data = 50;
	second->link = third;
	second->data = 20;
	first->link = second;
	first->data = 10;
	
	//del(&first, first, second);
	del(&first, trail, first);
	
	printf("List is: ");
	print_list(first);
	
	free(first);
	free(second);
	free(third);
	
	system("pause");
	return 0;
}
void del(list_pointer *ptr, list_pointer trail, list_pointer node)
/* trail is front of the deletes node ,node is want delete */
{
	if (trail)  // trail is Not NULL means node is not first node
		trail->link = node->link;
	else // trail is NULL means node is first node
		*ptr = (*ptr)->link;
}
void print_list(list_pointer ptr)
{
	for (; ptr; ptr = ptr->link)
		printf("%d ", ptr->data);
	printf("\n");
}

