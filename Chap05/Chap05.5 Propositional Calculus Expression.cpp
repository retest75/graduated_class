/* Chap05.5 Propositional Calculus Expression */ 
/* Expression Tree: (x1 and !x2) or (!x1 and x3) or !x3*/
//                     or(a)
//                /            \
//              or(b)         not(c)
//        /             \          \
//      and(d)         and(e)      x3(f)
//     /   \          /     \
//   x1(g) not(h)   not(i)  x3(j)
//             \        \
//             x2(k)    x1(l)

#include <stdio.h>
#include <stdlib.h>

typedef enum {NOT, AND, OR, TRUE, FALSE} logical;
typedef struct node node;
typedef node *node_pointer;
struct node
{
	node_pointer left;
	logical data;
	short int value;
	node_pointer right;
};

/* prototype */
node_pointer create(logical);
void free_tree(node_pointer);
void post_order(node_pointer);
void post_order_eval(node_pointer);

int main(void)
{
	node_pointer a, b, c, d, e, f, g, h, i, j, k, l;
	logical x1=TRUE, x2=FALSE, x3=TRUE;
	/* construct node */
	a = create(OR), b = create(OR),  c = create(NOT), d = create(AND), e = create(AND), f = create(x3);
	g = create(x1), h = create(NOT), i = create(NOT), j = create(x3),  k = create(x2),  l = create(x1);
	
	/* construct tree */
	a->left = b, a->right = c;                               // first layer
	b->left = d, b->right = e, c->right = f;                 // second layer
	d->left = g, d->right = h, e->left = i, e->right = j;    // third layer
	h->right = k, i->right = l;                              // fourth layer
	printf("When x1 = %d, x2 = %d, x3 = %d\n", x1, x2, x3);  // 3->True, 4->False
	
	/* post order to show expression tree */
	printf("Post Order of expression: ");
	post_order(a);  // x1 x2 NOT AND x1 NOT x3 AND OR x3 NOT OR
	printf("\n");
	
	/* calculus */
	post_order_eval(a);
	printf("Result of expression: ");
	printf("%d\n", a->value);  // 3->True, 4->False
	
	
	
	free_tree(a);
	system("pause");
	return 0;
}

node_pointer create(logical n)
{
	node_pointer temp;
	
	temp = (node_pointer) malloc(sizeof(node));
	temp->left = NULL;
	temp->data = n;
	temp->right = NULL;
	
	return temp;
}
void free_tree(node_pointer root)
{
	if (root)
	{
		free_tree(root->left);
		free_tree(root->right);
		free(root);
	}
}
void post_order(node_pointer root)
{
	if (root)
	{
		post_order(root->left);
		post_order(root->right);
		switch (root->data)
		{
			case NOT:
				printf("Not ");
				break;
			case AND:
				printf("And ");
				break;
			case OR:
				printf("Or ");
				break;
			case TRUE:
				printf("True ");
				break;
			case FALSE:
				printf("False ");
				break;
		}
    }
}
void post_order_eval(node_pointer root)
{
	if (root)
	{
		post_order_eval(root->left);
		post_order_eval(root->right);
		switch (root->data)
		{
			case NOT:
				root->value = !root->right->value;
				break;
			case AND:
				root->value = root->right->value && root->left->value;
				break;
			case OR:
				root->value = root->right->value || root->left->value;
				break;
			case TRUE:
				root->value = 1;
				break;
			case FALSE:
				root->value = 0;
				break;
		}
	}
}
 
