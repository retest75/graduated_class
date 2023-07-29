/* Chap04.3 Insert Node(difficult) */ 
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
list_pointer create_node(int);
list_pointer create_two_node(int, int);
void insert_node(list_pointer *, list_pointer, list_pointer);

int main(void)
{
	list_pointer ptr, node;
	
	ptr = create_two_node(10, 20);
	printf("-----Original list-----\n");
	printf("first node data:  %d\n", ptr->data);
	printf("first node link: %p\n", ptr->link);
	printf("second node data: %d\n", ptr->link->data);
	printf("second node link: %p\n", ptr->link->link);
	printf("\n");
	
	node = create_node(50);
	printf("-----inserted node-----\n");
	printf("inserted node data:  %d\n", node->data);
	printf("inserted node link: %p\n", node->link);
	printf("inserted node address: %p\n", &node);
	printf("\n");
	
	insert_node(&ptr, ptr, node);
	printf("-----after insert-----\n");
	printf("first node data:  %d\n", ptr->data);
	printf("first node link: %p\n", ptr->link);
	printf("second node data: %d\n", ptr->link->data);
	printf("second node link: %p\n", ptr->link->link);
	printf("third node data: %d\n", ptr->link->link->data);
	printf("third node link: %p\n", ptr->link->link->link);
	
	free(ptr->link->link); // free third node
	free(ptr->link);       // free second node
	free(ptr);             // free first node
	free(node);
	
	system("pause");
	return 0;
}

list_pointer create_node(int n)
{
	list_pointer Node;
	
	Node = (list_pointer) malloc(sizeof(list_node));
	Node->link = NULL;
	Node->data = n;
	
	return Node;
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
void insert_node(list_pointer *ptr, list_pointer pre_node, list_pointer insert_node)
/* insert a new node insert_node into the list ptr after pre_node */
{	
	if (*ptr) // Not NULL
	{
		insert_node->link = pre_node->link;
		pre_node->link = insert_node;
	}
	else      // NULL
		*ptr = insert_node;
}

