/* Chap05.1 Binary Tree Representation: Linked */
#include <stdio.h>
#include <stdlib.h>

typedef struct node node;
typedef node *tree_pointer;
struct node
{
	int data;
	tree_pointer left_child, right_child;
};

int main(void)
{
	int a=84, b=11;
	tree_pointer root, left_child, right_child;
	
	root = (tree_pointer) malloc(sizeof(node));
	left_child = (tree_pointer) malloc(sizeof(node));
	
	root->data = a;
	root->left_child = left_child;
	root->right_child = NULL;
	
	left_child->data = b;
	left_child->left_child = NULL;
	left_child->right_child = NULL;
	
	printf("root data: %d\n", root->data);
	printf("left node of root: %p\n", root->left_child);
	printf("right node of root: %p\n", root->right_child);
	
	printf("left child data: %d\n", root->left_child->data);
	printf("left node of left child: %p\n", root->left_child->left_child);
	printf("right child of left child: %p\n", root->left_child->right_child);
	
	system("pause");
	return 0;
	
}
