/* Chap05.2 Binary Tree ADT */
#include <stdio.h>
#include <stdlib.h>

/* define node */
typedef struct node node;
typedef node *tree_pointer;
struct node
{
	int data;
	tree_pointer left_child, right_child;
};

/* define operation function */
tree_pointer Create();
int Is_Empty(tree_pointer);


int main(void)
{
	tree_pointer root;
	int root_data, left_data, right_data;
	
	root = (tree_pointer) malloc(sizeof(node));
	root = Create();
	//printf("%d\n", root->data);
	//printf("%p\n", root->left_child);
	root_data = Is_Empty(root);
	left_data = Is_Empty(root->left_child);
	right_data = Is_Empty(root->right_child);
	printf("%d %d %d\n", root_data, left_data, right_data);
	
	
	system("pause");
	return 0;
}

tree_pointer Create()
{
	tree_pointer temp;
	
	temp = (tree_pointer) malloc(sizeof(node));
	temp->data = 0;
	temp->left_child = NULL;
	temp->right_child = NULL;
	
	return temp;
}
int Is_Empty(tree_pointer ptr)
{
	if (ptr==NULL)
		return 1;
	else
		return 0;
}
