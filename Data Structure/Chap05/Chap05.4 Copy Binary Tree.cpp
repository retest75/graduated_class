/* Chap05.4 Copy Binary Tree */

/* original binary tree */
//                a:10
//              /     \
//            b:20   c:30
//          /
//        d:40
//      /
//    e:50

#include <stdio.h>
#include <stdlib.h>
#define IS_FULL(ptr) ((ptr) == NULL)

typedef struct node node;
typedef node *node_pointer;
struct node
{
	int data;
	node_pointer left_node, right_node;
};

/* prototype */
node_pointer create(int);
node_pointer copy(node_pointer);
void preorder(node_pointer);
void free_tree(node_pointer);

int main(void)
{
	node_pointer a, b, c, d, e;
	node_pointer null_tree;
	node_pointer copy_tree1, copy_tree2;
	/* create originl tree */
	a = create(10), b = create(20), c = create(30), d = create(40), e = create(50); // create 5 nodes
	a->left_node = b, a->right_node = c;  // second layer
	b->right_node = d;                    // third layer
	d->left_node = e;                     // fourth layer
	printf("original tree: ");
	preorder(a);
	printf("\n");
	
	/* copy tree */
	copy_tree1 = copy(a);
	printf("duplicate tree: ");
	preorder(copy_tree1);
	printf("\n");
	
	null_tree = create(-1);
	copy_tree2 = copy(null_tree);
	if (copy_tree2 == NULL)
		printf("NULL tree !\n");
	else
		preorder(copy_tree2);
	
	free_tree(a);
	free_tree(null_tree);
	system("pause");
	return 0;
	
	
}

node_pointer create(int data)
{
	if (data == -1)
		return NULL;
	else
	{
		node_pointer temp;
		
		temp = (node_pointer) malloc(sizeof(node));
		temp->data = data;
		temp->left_node = NULL;
		temp->right_node = NULL;
		return temp;
	}
}
node_pointer copy(node_pointer original)
{
	if (original)
	{
		node_pointer temp;
		
		temp = (node_pointer) malloc(sizeof(node));
		if (IS_FULL(temp))
		{
			fprintf(stderr, "the memory is full !\n");
			exit(1);
		}
		/* use postorder to copy tree */
		temp->left_node = copy(original->left_node);   // L
		temp->right_node = copy(original->right_node); // R
		temp->data = original->data;                   // V
		return temp;
	}
	else
		return NULL;
}
void preorder(node_pointer root)
{
	if (root)
	{
		printf("%d ", root->data);
		preorder(root->left_node);
		preorder(root->right_node);
	}
}
void free_tree(node_pointer root)
{
	if (root)
	{
		free_tree(root->left_node);
		free_tree(root->right_node);
		free(root);
	}
}
