/* Chap04.3 Insert Node(easy) */ 
#include <stdio.h>
#include <stdlib.h>
#define IS_FULL(ptr) (!(ptr))

typedef struct list_node list_node;
typedef struct list_node *list_pointer;
struct list_node
{
	int data;
	list_pointer link;
};

/* prototype */
list_pointer create_two_node(int, int);
void insert_node(list_pointer *, list_pointer);

int main(void)
{
	list_pointer ptr;
	
	ptr = create_two_node(10, 20);
	insert_node(&ptr, ptr);
	printf("first node data:  %d\n", ptr->data);
	printf("first node link: %p\n", ptr->link);
	printf("second node data: %d\n", ptr->link->data);
	printf("second node link: %p\n", ptr->link->link);
	printf("third node data: %d\n", ptr->link->link->data);
	printf("third node link: %p\n", ptr->link->link->link);
	
	free(ptr->link->link); // free third node
	free(ptr->link);       // free second node
	free(ptr);             // free first node
	
	system("pause");
	return 0;
}

list_pointer create_two_node(int n, int m)
{
	list_pointer first, second;
	
	first = (list_pointer) malloc(sizeof(list_node));
	second = (list_pointer) malloc(sizeof(list_node));
	
	second->link = NULL;
	second->data = m;
	first->link = second;
	first->data = n;

	return first;
}
void insert_node(list_pointer *ptr, list_pointer pre_node)
/* insert a new node with data=50 into the list ptr after pre_node */
{
	list_pointer temp;
	
	temp = (list_pointer) malloc(sizeof(list_node)); /* if memory is full, then return NULL */
	if (IS_FULL(temp))
	{
		fprintf(stderr, "The memory is FULL\n");
		exit(1);
	}
	temp->data = 50;
	
	if (*ptr) // Not NULL
	{
		temp->link = pre_node->link;
		pre_node->link = temp;
	}
	else      // NULL
	{
		*ptr = temp;
		temp->link = NULL;
	}
}

