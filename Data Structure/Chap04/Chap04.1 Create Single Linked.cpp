/* Chap04.1 Create Single Linked */ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IS_EMPTY(ptr) (!(ptr))

typedef struct list_node list_node; // define a new type list_node
typedef list_node *list_pointer; // define a new type list_pointer, which is a pointer type, that point list_node
struct list_node
{
	char data[4];
	list_pointer link;
};

int main(void)
{
	/* because type list_pointer is a pointer type which point a node */
	/* if want to create a node, that means create a list_pointer */
	list_pointer empty_ptr=NULL; // declare a empty Node
	list_pointer Node;           // declare a Node
	
	/* create NULL node */
	if (IS_EMPTY(empty_ptr))
		printf("NULL\n");
	else
		printf("Not NULL\n");
	
	/* create one node having data */
	Node = (list_pointer) malloc(sizeof(list_node)); //allocation a memory for Node
	strcpy(Node->data, "bat");
	Node->link = NULL;
	if (IS_EMPTY(Node))
		printf("NULL\n");
	else
	{
		printf("Not NULL\n");
		printf("%s\n", Node->data);
	}
	free(Node);
	
	system("pause");
	return 0;
}

