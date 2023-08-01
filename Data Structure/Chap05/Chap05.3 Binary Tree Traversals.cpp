/* Chap05.3 Binary Tree Traversals */
#include <stdio.h>
#include <stdlib.h>

/* binary tree */
//                a:10
//              /     \
//            b:20   c:30
//          /
//        d:40
//      /
//    e:50

typedef struct node node;
typedef node *node_pointer;
struct node
{
	int data;
	node_pointer left_node, right_node;
};

/* prototype */
node_pointer Create(int);     // create node
void freeTree(node_pointer);  // release memory space
void inorder(node_pointer);
void preorder(node_pointer);
void postorder(node_pointer);

int main(void)
{
	node_pointer a, b, c, d, e;
	
	/* cerate node and binary tree */
	a = Create(10); // root
	b = Create(20); // left node of a
	c = Create(30); // right node of a
	d = Create(40); // left node of b
	e = Create(50); // left node of d
	a->left_node = b;
	a->right_node = c;
	b->left_node = d;
	d->left_node = e;
	
	/* traversal */
	printf("Inorder traversal: ");
	inorder(a);   // 50 40 20 10 30
	printf("\n");
	
	printf("Preorder traversal: ");
	preorder(a);  // 10 20 40 50 30
	printf("\n");
	
	printf("Postprdef traversal: ");
	postorder(a); // 50 40 20 30 10
	printf("\n");
	
	freeTree(a);
	system("pause");
	return 0;
}

node_pointer Create(int n)
{
	node_pointer temp;
	
	temp = (node_pointer) malloc(sizeof(node));
	temp->data = n;
	temp->left_node = NULL;
	temp->right_node = NULL;
	
	return temp;
}
void freeTree(node_pointer ptr)
{
    if (ptr)
    {
        freeTree(ptr->left_node);
        freeTree(ptr->right_node);
        free(ptr);
    }
}
void inorder(node_pointer ptr)
{
	if (ptr)
	{
		inorder(ptr->left_node);   // L
		printf("%d ", ptr->data);  // V
		inorder(ptr->right_node);  // R
	}
}
void preorder(node_pointer ptr)
{
	if (ptr)
	{
		printf("%d ", ptr->data);  // V
		preorder(ptr->left_node);  // L
		preorder(ptr->right_node); // R
	}
}
void postorder(node_pointer ptr)
{
	if (ptr)
	{
		postorder(ptr->left_node);   // L
		postorder(ptr->right_node);  // R
		printf("%d ", ptr->data);    // V
	}
}
