/* Chap03.2 Queue ADT */

#include <stdio.h> 
#include <stdlib.h>
#define MAX_QUEUE_SIZE 5

typedef struct
{
	int key; // other fields
} element;

/* create a global Queue */
element queue[MAX_QUEUE_SIZE];
int rear, front = -1;

/* operation function */
int IsFull(int);
int IsEmpty(int, int);
void push(int *, element);
element pop(int *, int);

/* show function */
void print_queue(element *);

int main(void)
{
	element a = {900607}, item;
	queue[0].key = 84;
	queue[1].key = 11;
	queue[2].key = 20;
	rear = 2;
	front = 0;
	
	printf("Is queue is full ? ");
	if (IsFull(rear))
		printf("Full !\n");
	else
		printf("Not Full !\n");
	
	printf("Is queue is empty ? ");
	if (IsEmpty(rear, front))
		printf("Empty !\n");
	else
		printf("Not Empty !\n");
	printf("----------\n");
	
	printf("before push, rear = %d, front = %d\n", rear, front);
	push(&rear, a);
	printf("after push, rear = %d, front = %d\n", rear, front);
	print_queue(queue);
	printf("----------\n");
	
	printf("before pop, rear = %d, front = %d\n", rear, front);
	item = pop(&front, rear);
	printf("after pop, rear = %d, front = %d\n", rear, front);
	printf("element you just delete is : %d\n", item.key);
	print_queue(queue);
	
	system("pause");
	return 0;
}
int IsFull(int rear)
{
	if (rear == MAX_QUEUE_SIZE-1) // full
		return 1;
	else 
		return 0;
}
int IsEmpty(int rear, int front)
{
	if (rear == front) // empty
		return 1;
	else
	 return 0;
}
void push(int *rear_ptr, element item)
{
	if (*rear_ptr == MAX_QUEUE_SIZE-1)
		printf("Queue is full\n");
	else
		queue[++(*rear_ptr)] = item;
}
element pop(int *front_ptr, int rear) /* return element you delete */
{
	element empty_element = {0};
	if (*front_ptr == rear)
	{
		printf("Queue is empty\n");
		return empty_element;
	}
	else
		return queue[(*front_ptr)++];
}
void print_queue(element *queue_ptr)
{
	int i;
	
	printf("rear = %d, front = %d, Queue = [ ", rear, front);
	for (i=front; i<=rear; i++)
		printf("%d ", (queue_ptr+i)->key);
	printf("]\n");
}
